"""
api_bridge.py v7 — mirrors EXACTLY what the CLI does
=====================================================
CLI flow for "Warfarin + Aspirin":
  mock_ocr = {brand_name:"Warfarin", generic_name:"Warfarin", ...}
  identity = resolve_drug(mock_ocr)          → {canonical_name:"warfarin", ...}
  intr     = check_interactions(identity, ["Aspirin"])
  report   = build_report(identity, intr, source_mode="text")

This bridge does EXACTLY that for every pair.
"""

import os, sys, uuid, shutil, traceback, itertools
from datetime import datetime, timedelta
from pathlib  import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security        import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles     import StaticFiles
from pydantic import BaseModel

import bcrypt
from jose           import jwt, JWTError
from pymongo        import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError
from bson           import ObjectId

# ── Find core/ ───────────────────────────────────────────────
def _find_core() -> Path:
    for p in [Path(__file__).parent/"core",
              Path(__file__).parent.parent/"core",
              Path.cwd()/"core"]:
        if p.is_dir() and (p/"resolver.py").exists():
            return p
    raise RuntimeError("Cannot find core/. Run uvicorn from the folder containing core/")

CORE_DIR = _find_core()
print(f"[PharmaAI] core/ → {CORE_DIR}")
sys.path.insert(0, str(CORE_DIR))

from resolver     import resolve_drug
from interactions import check_interactions
from report       import build_report
print("[PharmaAI] Pipeline OK")

# ── Config ───────────────────────────────────────────────────
SECRET_KEY    = "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"
ALGORITHM     = "HS256"
ACCESS_EXPIRE = 120   # 2 hours — avoids token expiry during long analysis

BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = str(BASE_DIR / "uploads")
MONGO_URI  = "mongodb://localhost:27017"
MONGO_DB   = "pharma_ai"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(str(BASE_DIR / "reports"), exist_ok=True)

# ── MongoDB ──────────────────────────────────────────────────
try:
    _mc = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    _mc.admin.command("ping")
    print("[PharmaAI] MongoDB OK")
except Exception as e:
    raise RuntimeError(f"MongoDB: {e}")

mdb        = _mc[MONGO_DB]
users_col  = mdb["users"]
checks_col = mdb["drug_checks"]
fraud_col  = mdb["fraud_log"]
ip_col     = mdb["ip_counters"]

users_col.create_index("email", unique=True)
checks_col.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
ip_col.create_index([("ip_address", ASCENDING), ("window_start", ASCENDING)], unique=True)

_apw = bcrypt.hashpw(b"Admin@123", bcrypt.gensalt()).decode()
users_col.update_one({"email": "admin@pharma.ai"},
    {"$setOnInsert": {"full_name":"Admin","email":"admin@pharma.ai",
                      "password_hash":_apw,"role":"admin",
                      "is_active":True,"created_at":datetime.utcnow()}}, upsert=True)

# ════════════════════════════════════════════════════════════
#  KEY READING — from your backend terminal output we know:
#
#  resolve_drug() returns a flat dict with keys:
#    input_name, canonical_name, brand_name, generic_name,
#    drug_class, rxcui, pubchem_cid, mol_formula, mol_weight,
#    smiles, route, product_type, manufacturer_fda,
#    suspicion_score, suspicion_level, authenticity_verdict,
#    suspicion_flags
#
#  check_interactions() returns:
#    interactions (list), summary, overall_risk,
#    drugs_checked, drugs_unresolved
#
#  Each interaction item:
#    drug_a, drug_b, severity, description, source,
#    all_sources, tier
#
#  build_report() returns:
#    meta, drug_identity, authenticity, interactions, combined_risk
#  Where:
#    report["interactions"]["detail"]  ← the interaction list
#    report["combined_risk"]["level"]  ← CRITICAL/HIGH/MODERATE/LOW/NONE
#    report["authenticity"]["verdict"] ← "⚠️  WARN" etc
#    report["authenticity"]["score"]   ← 0.0-1.0
# ════════════════════════════════════════════════════════════

def _make_mock_ocr(name: str) -> dict:
    """Exact same mock_ocr the CLI uses for text mode."""
    safe = (name or "").strip() or "Unknown"   # never None or empty
    return {
        "success":       True,
        "_source_mode":  "text",
        "brand_name":    safe,
        "generic_name":  safe,
        "dosage":        None,
        "dosage_form":   None,
        "batch_no":      None,
        "mfg_date":      None,
        "exp_date":      None,
        "manufacturer":  None,
        "license_no":    None,
        "storage":       None,
        "raw_text":      safe,
    }

def _clean_verdict(raw: str) -> str:
    s = str(raw).replace("✅","").replace("🚨","").replace("⚪","").replace("⚠️","").replace("⚠","").strip().upper()
    if "AUTHENTIC" in s:  return "AUTHENTIC"
    if "WARN" in s:       return "WARN"
    if "HIGH" in s:       return "SUSPICIOUS"
    if "MODERATE" in s:   return "WARN"
    return s or "UNKNOWN"

def _rec(level: str) -> str:
    return {
        "CRITICAL": "🚫 CRITICAL: Do NOT use this combination. Consult a physician immediately.",
        "HIGH":     "🔴 HIGH RISK: Consult a pharmacist or physician before taking this combination.",
        "MODERATE": "🟡 MODERATE RISK: Use with caution. Monitor for adverse effects.",
        "LOW":      "🟢 LOW RISK: Minor concerns noted. Follow prescribing instructions.",
        "NONE":     "✅ No significant interactions detected.",
    }.get(level.upper(), "Consult a healthcare professional.")

def _shape_drug(identity: dict, label: str, report: dict) -> dict:
    """
    Extract display fields from resolve_drug() output.
    Uses exact keys we see in the terminal output.
    """
    auth = report.get("authenticity", {})
    di   = report.get("drug_identity", {})
    return {
        "label":           label,
        "brand_name":      di.get("brand_name")   or identity.get("brand_name")  or label,
        "generic_name":    di.get("generic_name")  or identity.get("canonical_name") or identity.get("generic_name") or "",
        "canonical_name":  identity.get("canonical_name") or "",
        "rxcui":           identity.get("rxcui")   or "",
        "drug_class":      identity.get("drug_class") or "",
        "dosage_form":     di.get("dosage_form")   or "",
        "dosage":          di.get("dosage")        or "",
        "route":           identity.get("route")   or "",
        "formula":         identity.get("mol_formula") or identity.get("smiles") or "",
        "mol_weight":      str(identity.get("mol_weight") or ""),
        "pubchem_cid":     str(identity.get("pubchem_cid") or ""),
        "manufacturer":    identity.get("manufacturer_fda") or di.get("manufacturer") or "",
        # auth — from terminal: score=0.15 means suspicion_score=0.15
        "auth_score":      auth.get("score") or identity.get("suspicion_score") or 0.0,
        "auth_level":      auth.get("level") or identity.get("suspicion_level") or "UNKNOWN",
        "verdict":         _clean_verdict(auth.get("verdict") or identity.get("authenticity_verdict") or "UNKNOWN"),
        "suspicion_flags": auth.get("flags") or identity.get("suspicion_flags") or [],
    }

def _shape_pair(intr_result: dict, a_label: str, b_label: str) -> dict:
    """
    Map check_interactions() output to display structure.
    interactions.py returns 'interactions' list where each item has:
    drug_a, drug_b, severity, description, source, all_sources
    """
    raw_list = intr_result.get("interactions", [])
    items = []
    for x in raw_list:
        if not isinstance(x, dict):
            continue
        items.append({
            "drug_a":      x.get("drug_a", a_label),
            "drug_b":      x.get("drug_b", b_label),
            "severity":    x.get("severity", "unknown"),
            "description": x.get("description", ""),
            "source":      ", ".join(x.get("all_sources", [x.get("source", "")])),
        })

    return {
        "drug_a":       a_label,
        "drug_b":       b_label,
        "overall_risk": intr_result.get("overall_risk", "NONE"),
        "summary":      intr_result.get("summary", {}),
        "interactions": items,
        "drugs_checked":intr_result.get("drugs_checked", []),
        "unresolved":   intr_result.get("drugs_unresolved", []),
    }

# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="PharmaAI", version="7.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
bearer_scheme = HTTPBearer(auto_error=False)

# ── Debug (no auth needed) ───────────────────────────────────
@app.get("/api/debug")
def debug():
    """Run Warfarin+Aspirin through the real pipeline and show raw output."""
    try:
        ocr_w = _make_mock_ocr("Warfarin")
        ocr_a = _make_mock_ocr("Aspirin")
        id_w  = resolve_drug(ocr_w)
        id_a  = resolve_drug(ocr_a)
        # EXACTLY as CLI: check_interactions(warfarin_identity, ["Aspirin"])
        intr  = check_interactions(id_w, ["Aspirin"])
        rep   = build_report(id_w, intr, source_mode="text")
        return {
            "status":             "OK",
            "identity_keys":      list(id_w.keys()),
            "report_keys":        list(rep.keys()),
            "interaction_count":  len(intr.get("interactions", [])),
            "overall_risk":       intr.get("overall_risk"),
            "first_interaction":  intr.get("interactions", [None])[0],
            "identity_warfarin":  id_w,
            "report_structure":   {k: list(v.keys()) if isinstance(v,dict) else type(v).__name__
                                   for k,v in rep.items()},
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e), "trace": traceback.format_exc()}

# ── JWT ───────────────────────────────────────────────────────
def _make_token(data, exp): return jwt.encode({**data,"exp":datetime.utcnow()+exp}, SECRET_KEY, algorithm=ALGORITHM)
def _decode(tok):
    try:    return jwt.decode(tok, SECRET_KEY, algorithms=[ALGORITHM])
    except: raise HTTPException(401, "Invalid or expired token")
def get_user(creds: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if not creds: raise HTTPException(401, "Not authenticated")
    p = _decode(creds.credentials)
    return {"id":p["sub"],"email":p["email"],"role":p["role"]}

class RegBody(BaseModel): full_name:str; email:str; password:str
class LoginBody(BaseModel): email:str; password:str

@app.post("/api/auth/register")
def register(b: RegBody):
    pw = bcrypt.hashpw(b.password.encode(), bcrypt.gensalt()).decode()
    try:
        r = users_col.insert_one({"full_name":b.full_name,"email":b.email.lower().strip(),
            "password_hash":pw,"role":"user","is_active":True,"created_at":datetime.utcnow()})
    except DuplicateKeyError: raise HTTPException(409,"Email already registered")
    return {"access_token":_make_token({"sub":str(r.inserted_id),"email":b.email,"role":"user"},
             timedelta(minutes=ACCESS_EXPIRE)), "role":"user","full_name":b.full_name}

@app.post("/api/auth/login")
def login(b: LoginBody):
    u = users_col.find_one({"email":b.email.lower().strip(),"is_active":True})
    if not u or not bcrypt.checkpw(b.password.encode(), u["password_hash"].encode()):
        raise HTTPException(401,"Invalid credentials")
    return {"access_token":_make_token({"sub":str(u["_id"]),"email":u["email"],"role":u["role"]},
             timedelta(minutes=ACCESS_EXPIRE)), "role":u["role"],"full_name":u["full_name"]}

@app.get("/api/auth/me")
def me(user=Depends(get_user)): return user

# ════════════════════════════════════════════════════════════
#  /api/analyze  — pairwise interaction
#
#  Accepts: drug_names (newline-separated string) + optional images
#  For N drugs → N*(N-1)/2 pairs checked
# ════════════════════════════════════════════════════════════
@app.post("/api/analyze")
async def analyze(
    request:    Request,
    images:     list[UploadFile] = File(default=[]),
    drug_names: str = Form(default=""),
    user = Depends(get_user)
):
    print(f"\n[PharmaAI] /api/analyze — drug_names={repr(drug_names)}")

    real_images = [img for img in images if img.filename]
    name_list   = [n.strip() for n in drug_names.replace(",","\n").split("\n") if n.strip()]

    print(f"[PharmaAI] images={len(real_images)}  text_names={name_list}")

    if not real_images and not name_list:
        raise HTTPException(400, "Provide at least one drug name or image")

    # ── STEP 1: Resolve each drug ─────────────────────────────
    # Each entry: {label, identity, mock_ocr}
    resolved = []
    errors   = []

    # From images
    if real_images:
        try:
            from ocr_fixed import run_ocr as _run_ocr_raw
        except ImportError:
            from ocr import run_ocr as _run_ocr_raw

        def safe_ocr(path, filename):
            """
            Wraps run_ocr and catches the NoneType.strip() crash that happens
            when OpenRouter returns null content (rate-limit / model issue on free tier).
            Also ensures brand_name/generic_name are never None before resolve_drug.
            """
            try:
                result = _run_ocr_raw(path)
            except AttributeError as e:
                if "strip" in str(e) or "NoneType" in str(e):
                    return {"success": False,
                            "error": "OpenRouter API returned null content — the free model is "
                                     "rate-limited. Wait 30 sec and retry, or check your API key."}
                return {"success": False, "error": str(e)}
            except Exception as e:
                return {"success": False, "error": str(e)}

            if not isinstance(result, dict):
                return {"success": False, "error": "OCR returned unexpected type"}

            # Catch when crash is wrapped inside success=False error string
            err_str = str(result.get("error", ""))
            if not result.get("success") and ("NoneType" in err_str or "'strip'" in err_str):
                return {"success": False,
                        "error": "OpenRouter returned null — model rate-limited. "
                                 "Wait 30 sec and retry, or use a clearer image."}

            if not result.get("success"):
                return result

            # Guarantee no None fields — resolver._clean_name() calls .strip() on these
            stem = Path(filename).stem or "drug"
            bn = result.get("brand_name") or result.get("generic_name") or stem
            gn = result.get("generic_name") or bn
            result["brand_name"]   = str(bn).strip() if bn else stem
            result["generic_name"] = str(gn).strip() if gn else stem
            result["raw_text"]     = result.get("raw_text") or stem
            result["_source_mode"] = "image"
            return result

        for img_file in real_images:
            ext  = Path(img_file.filename).suffix or ".jpg"
            dest = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
            with open(dest, "wb") as fh:
                shutil.copyfileobj(img_file.file, fh)
            try:
                ocr = safe_ocr(dest, img_file.filename)
                if not ocr.get("success"):
                    errors.append({"label": img_file.filename, "error": ocr.get("error", "OCR failed")})
                    continue
                identity = resolve_drug(ocr)
                print(f"[PharmaAI] Resolved image '{img_file.filename}' → '{identity.get('canonical_name')}'")
                resolved.append({"label": ocr["brand_name"], "identity": identity, "ocr": ocr})
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[PharmaAI] Image error: {e}\n{tb}")
                errors.append({"label": img_file.filename, "error": str(e)})

    # From text names — mirrors CLI exactly
    for name in name_list:
        try:
            mock_ocr = _make_mock_ocr(name)
            identity = resolve_drug(mock_ocr)
            print(f"[PharmaAI] Resolved '{name}' → canonical='{identity.get('canonical_name')}'  rxcui={identity.get('rxcui')}")
            resolved.append({"label": name, "identity": identity, "ocr": mock_ocr})
        except Exception as e:
            errors.append({"label":name,"error":str(e),"trace":traceback.format_exc()})

    if not resolved:
        return {"drugs":[],"pairs":[],"combined":{},"errors":errors,
                "debug":"All drugs failed to resolve — check uvicorn log"}

    # ── STEP 2: Pairwise interactions ─────────────────────────
    # For pair (A, B):
    #   check_interactions(A.identity, [B.label])
    # This mirrors: check_interactions(warfarin_identity, ["Aspirin"])
    RANK = {"CRITICAL":4,"HIGH":3,"MODERATE":2,"LOW":1,"NONE":0,"UNKNOWN":0}
    pairs       = []
    worst_risk  = "NONE"
    total_iact  = 0

    for i, j in itertools.combinations(range(len(resolved)), 2):
        a = resolved[i]
        b = resolved[j]
        a_label = a["label"]
        b_label = b["label"]

        # Pass drug B's label as the "other drug" string — same as CLI "Other medications: Aspirin"
        print(f"[PharmaAI] Checking pair: '{a_label}' ↔ '{b_label}'")
        try:
            intr_result = check_interactions(a["identity"], [b_label])
            print(f"[PharmaAI] → {len(intr_result.get('interactions',[]))} interactions, risk={intr_result.get('overall_risk')}")
            pair = _shape_pair(intr_result, a_label, b_label)
            pairs.append(pair)
            total_iact += len(pair["interactions"])
            if RANK.get(pair["overall_risk"],0) > RANK.get(worst_risk,0):
                worst_risk = pair["overall_risk"]
        except Exception as e:
            print(f"[PharmaAI] Pair error: {e}")
            pairs.append({"drug_a":a_label,"drug_b":b_label,
                           "overall_risk":"UNKNOWN","interactions":[],
                           "error":str(e),"trace":traceback.format_exc()})

    # ── STEP 3: Build individual drug reports + shape ─────────
    shaped_drugs = []
    for entry in resolved:
        try:
            # Build a solo report (no interactions) just to get auth/identity fields
            empty_intr = {"interactions":[],"summary":{},"overall_risk":"NONE",
                          "drugs_checked":[],"drugs_unresolved":[]}
            solo_report = build_report(entry["identity"], empty_intr, source_mode="text")
            shaped_drugs.append(_shape_drug(entry["identity"], entry["label"], solo_report))
        except Exception as e:
            print(f"[PharmaAI] shape drug error for {entry['label']}: {e}")
            shaped_drugs.append({"label":entry["label"],"brand_name":entry["label"],
                                  "error":str(e)})

    # ── STEP 4: Combined verdict ──────────────────────────────
    combined = {
        "overall_interaction_risk": worst_risk,
        "total_interactions":       total_iact,
        "drug_count":               len(shaped_drugs),
        "pair_count":               len(pairs),
        "recommendation":           _rec(worst_risk),
        "any_auth_issue":           any(d.get("verdict") not in ("AUTHENTIC","UNKNOWN") for d in shaped_drugs),
    }

    print(f"[PharmaAI] Done — worst_risk={worst_risk}  total_interactions={total_iact}")

    # ── Save to DB ────────────────────────────────────────────
    try:
        drug_str = ", ".join(d["label"] for d in shaped_drugs)
        checks_col.insert_one({
            "user_id":user["id"],"input_mode":"multi","drug_name":drug_str,
            "drug_count":len(shaped_drugs),"interaction_count":total_iact,
            "risk_level":worst_risk,"verdict":"SUSPICIOUS" if combined["any_auth_issue"] else "OK",
            "ip_address":request.client.host,"created_at":datetime.utcnow()})
        _fraud(user["id"], request.client.host, drug_str)
    except Exception as e:
        print(f"[PharmaAI] DB save warning: {e}")

    return {
        "drugs":    shaped_drugs,
        "pairs":    pairs,
        "combined": combined,
        "errors":   errors,
    }

def _fraud(uid, ip, drug):
    try:
        w = datetime.utcnow().replace(minute=0,second=0,microsecond=0)
        k = {"ip_address":ip,"window_start":w}
        ex = ip_col.find_one(k)
        if ex:
            ds = set(ex.get("drug_set",[]))
            ds.add(drug or "")
            ip_col.update_one(k,{"$set":{"drug_set":list(ds)},"$inc":{"check_count":1}})
            if len(ds)>5:
                fraud_col.update_one({"ip_address":ip,"window_start":w},
                    {"$setOnInsert":{"user_id":uid,"ip_address":ip,"window_start":w,
                        "event_type":"burst","details":{"drugs":list(ds)},
                        "flagged_at":datetime.utcnow(),"reviewed":False}},upsert=True)
        else:
            ip_col.insert_one({**k,"drug_set":[drug or ""],"check_count":1})
    except: pass

# ── History / Stats / Fraud ───────────────────────────────────
@app.get("/api/history")
def history(limit:int=50, user=Depends(get_user)):
    rows = list(checks_col.find({"user_id":user["id"]}).sort("created_at",DESCENDING).limit(limit))
    for r in rows:
        r["id"]=str(r.pop("_id"))
        r["created_at"]=r["created_at"].isoformat() if r.get("created_at") else None
    return {"history":rows}

@app.get("/api/stats")
def stats(user=Depends(get_user)):
    uid=user["id"]
    def agg(grp,proj): return list(checks_col.aggregate([{"$match":{"user_id":uid}},{"$group":grp},{"$project":proj}]))
    verdicts = agg({"_id":"$verdict","count":{"$sum":1}},{"verdict":"$_id","count":1,"_id":0})
    risks    = agg({"_id":"$risk_level","count":{"$sum":1}},{"risk_level":"$_id","count":1,"_id":0})
    since    = datetime.utcnow()-timedelta(days=14)
    timeline = list(checks_col.aggregate([
        {"$match":{"user_id":uid,"created_at":{"$gte":since}}},
        {"$group":{"_id":{"$dateToString":{"format":"%Y-%m-%d","date":"$created_at"}},"count":{"$sum":1}}},
        {"$sort":{"_id":1}},{"$project":{"day":"$_id","count":1,"_id":0}}]))
    tp = list(checks_col.aggregate([{"$match":{"user_id":uid}},
        {"$group":{"_id":None,"total":{"$sum":1},"total_interactions":{"$sum":"$interaction_count"}}}]))
    totals = tp[0] if tp else {"total":0,"total_interactions":0}
    totals.pop("_id",None)
    return {"verdicts":verdicts,"risks":risks,"timeline":timeline,"totals":totals}

@app.get("/api/admin/fraud")
def fraud_log(user=Depends(get_user)):
    if user["role"]!="admin": raise HTTPException(403,"Admin only")
    rows=list(fraud_col.find().sort("flagged_at",DESCENDING).limit(100))
    for r in rows:
        r["id"]=str(r.pop("_id"))
        try:
            u=users_col.find_one({"_id":ObjectId(r["user_id"])},{"email":1})
            r["email"]=u["email"] if u else "—"
        except: r["email"]="—"
        r["flagged_at"]=r["flagged_at"].isoformat() if r.get("flagged_at") else None
        r["window_start"]=r["window_start"].isoformat() if r.get("window_start") else None
    return {"fraud_events":rows}