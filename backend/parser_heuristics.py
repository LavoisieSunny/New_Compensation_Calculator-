import re
from datetime import datetime

def parse_extracted_text(text_lines):
    """
    Parses OCR extracted text lines to find fields for the Compensation Calculator.
    Returns a dictionary of suggested form values by focusing on Previous Judgment, Petition, and Prayer sections.
    """
    # Combine lines for full text searches
    full_text = "\n".join(text_lines)
    full_text_lower = full_text.lower()

    suggestions = {
        "case_type": "injury", # default
        "name": "",
        "father_name": "",
        "date_of_accident": "",
        "date_of_birth": "",
        "age": "",
        "monthly_income": "",
        "disability": "",
        "dependents": "",
        "marital_status": "married", # default
        "award_amount": "", # parsed from final award section
        "place_of_accident": "", # parsed accident location
    }

    # ======================================================
    # 1. SECTION DIVISION (PETITION, PRAYER, PREVIOUS JUDGMENT)
    # ======================================================
    # Find offsets for sections to query them locally
    sections = {
        "petition": "",
        "prayer": "",
        "judgment": "",
        "general": full_text
    }
    
    petition_headers = ["petition under", "claim petition", "averments", "brief facts", "the claimant", "statement of facts", "petitioner"]
    prayer_headers = ["prayer", "prayed that", "relief claimed", "wherefore", "it is respectfully prayed", "reliefs"]
    judgment_headers = ["judgment", "award", "decision", "findings on issue", "held that", "court orders", "decree"]
    
    pet_idx = -1
    for h in petition_headers:
        idx = full_text_lower.find(h)
        if idx != -1:
            pet_idx = idx
            break
            
    pr_idx = -1
    for h in prayer_headers:
        idx = full_text_lower.find(h)
        if idx != -1:
            pr_idx = idx
            break
            
    jud_idx = -1
    for h in judgment_headers:
        idx = full_text_lower.find(h)
        if idx != -1:
            jud_idx = idx
            break
            
    # Sort indices to slice
    slices = sorted([("petition", pet_idx), ("prayer", pr_idx), ("judgment", jud_idx)], key=lambda x: x[1])
    
    for i in range(len(slices)):
        name, start_idx = slices[i]
        if start_idx == -1:
            continue
        end_idx = len(full_text)
        for j in range(i + 1, len(slices)):
            if slices[j][1] != -1:
                end_idx = slices[j][1]
                break
        sections[name] = full_text[start_idx:end_idx]

    # Combine localized text for targeted queries
    petition_text = (sections["petition"] or full_text).lower()
    prayer_text = (sections["prayer"] or full_text).lower()
    judgment_text = (sections["judgment"] or full_text).lower()

    # ======================================================
    # 2. CASE TYPE DETERMINATION
    # ======================================================
    death_keywords = [
        "death", "deceased", "fatal", "post mortem", "died on", 
        "consortium", "funeral", "loss of estate", "widow", "widower", 
        "claimant is wife", "claimants are children"
    ]
    # Check within judgment and prayer first
    death_score = sum(3 for kw in death_keywords if kw in judgment_text) + sum(1 for kw in death_keywords if kw in petition_text)
    
    injury_keywords = [
        "injury", "injured", "treatment", "disability certificate", 
        "fracture", "wounds", "medical expenses", "pain and suffering", 
        "discharge summary", "hospitalized"
    ]
    injury_score = sum(3 for kw in injury_keywords if kw in judgment_text) + sum(1 for kw in injury_keywords if kw in petition_text)
    
    if death_score > injury_score:
        suggestions["case_type"] = "death"
    else:
        suggestions["case_type"] = "injury"

    # ======================================================
    # 3. EXTRACT ACCIDENT PLACE (Focus on Petition & facts)
    # ======================================================
    place_patterns = [
        r'place\s+of\s+(?:accident|occurrence)\s*[:\-]\s*([A-Za-z0-9\s\.\,\-\/]{3,60})',
        r'accident\s+(?:occurred|took\s+place)\s+at\s+([A-Za-z0-9\s\.\,\-\/]{3,60})',
        r'accident\s+near\s+([A-Za-z0-9\s\.\,\-\/]{3,60})',
        r'occurred\s+near\s+([A-Za-z0-9\s\.\,\-\/]{3,60})',
        r'collision\s+at\s+([A-Za-z0-9\s\.\,\-\/]{3,60})',
        r'on\s+national\s+highway\s+no\.?\s*(\d+|\w+)',
        r'on\s+nh\s*[-–]?\s*(\d+|\w+)',
        r'at\s+village\s+([A-Za-z\s]{3,30})'
    ]
    
    for pat in place_patterns:
        # Search in petition text first (usually details the accident incident location)
        match = re.search(pat, petition_text)
        if not match:
            # Fallback to general text
            match = re.search(pat, full_text_lower)
        if match:
            place_val = match.group(1).strip()
            # Clean up trailing punctuation or garbage lines
            place_val = re.split(r'\n|\r|\.|,|claims', place_val)[0].strip()
            if len(place_val) > 3 and not any(stop in place_val.lower() for stop in ["is", "the", "date", "time", "rs"]):
                suggestions["place_of_accident"] = place_val.title()
                break

    # ======================================================
    # 4. EXTRACT DATES (Focus on Petition / General)
    # ======================================================
    # Standard Indian formats: DD-MM-YYYY or DD/MM/YYYY or DD.MM.YYYY
    date_pattern = r'\b(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{4})\b'
    dates_found = []
    
    for match in re.finditer(date_pattern, full_text):
        date_str = f"{int(match.group(1)):02d}-{int(match.group(2)):02d}-{match.group(3)}"
        start_idx = max(0, match.start() - 50)
        end_idx = min(len(full_text), match.end() + 50)
        context = full_text[start_idx:end_idx].lower()
        dates_found.append({"date": date_str, "context": context})

    accident_keywords = ["accident", "incident", "occurrence", "date of f.i.r", "fir", "clash", "crash", "collision", "happened on", "occurred on"]
    dob_keywords = ["dob", "date of birth", "born on", "birth", "d.o.b"]

    for d in dates_found:
        if any(kw in d["context"] for kw in dob_keywords):
            suggestions["date_of_birth"] = d["date"]
        elif any(kw in d["context"] for kw in accident_keywords):
            suggestions["date_of_accident"] = d["date"]

    # Deduce if missing but multiple dates found
    if len(dates_found) >= 2:
        parsed_dates = []
        for d in dates_found:
            try:
                parsed_dates.append(datetime.strptime(d["date"], "%d-%m-%Y"))
            except ValueError:
                pass
        
        if len(parsed_dates) >= 2:
            # Sort chronologically. DOB is always earlier than Accident Date!
            sorted_dates = [d[0] for d in sorted(zip(dates_found, parsed_dates), key=lambda x: x[1])]
            if not suggestions["date_of_birth"]:
                suggestions["date_of_birth"] = sorted_dates[0]["date"]
            if not suggestions["date_of_accident"]:
                suggestions["date_of_accident"] = sorted_dates[1]["date"]
    elif len(dates_found) == 1:
        d = dates_found[0]
        if any(kw in d["context"] for kw in dob_keywords):
            suggestions["date_of_birth"] = d["date"]
        else:
            suggestions["date_of_accident"] = d["date"]

    # ======================================================
    # 5. EXTRACT AGE (Judgment / General)
    # ======================================================
    age_patterns = [
        r'\bage\b\s*(?:of|is|:)?\s*(\d{1,2})\b',
        r'\baged\b\s*(?:about)?\s*(\d{1,2})\b',
        r'\b(\d{1,2})\s*years\s*(?:old)?\b'
    ]
    # Check inside Judgment/Previous Judgment first
    for pattern in age_patterns:
        matches = re.findall(pattern, judgment_text)
        if not matches:
            matches = re.findall(pattern, full_text_lower)
        if matches:
            suggestions["age"] = int(matches[0])
            break

    # ======================================================
    # 6. EXTRACT MONTHLY INCOME (Judgment / Prayer)
    # ======================================================
    income_patterns = [
        r'(?:monthly|salary|income|earnings?|wage?s?)\b.*?\b(?:rs\.?|inr|rupees)?\s*(\d{4,6})\b',
        r'\b(?:rs\.?|inr)\s*(\d{4,6})\b\s*(?:per\s*month|\/pm|\/-\s*pm|p\.m\.)'
    ]
    for pattern in income_patterns:
        # Check inside judgment/prayer first
        matches = re.findall(pattern, judgment_text)
        if not matches:
            matches = re.findall(pattern, prayer_text)
        if not matches:
            matches = re.findall(pattern, full_text_lower)
        if matches:
            suggestions["monthly_income"] = float(matches[0])
            break

    # ======================================================
    # 7. EXTRACT PERMANENT DISABILITY (Judgment / General)
    # ======================================================
    disability_patterns = [
        r'\b(\d{1,2}(?:\.\d+)?)\s*%\s*(?:permanent)?\s*disability\b',
        r'\bdisability\b\s*(?:of|is|:)?\s*(\d{1,2}(?:\.\d+)?)\s*%',
        r'\bpermanent disability\b\s*(?:of|is|:)?\s*(\d{1,2}(?:\.\d+)?)\s*%'
    ]
    for pattern in disability_patterns:
        matches = re.findall(pattern, judgment_text)
        if not matches:
            matches = re.findall(pattern, full_text_lower)
        if matches:
            suggestions["disability"] = float(matches[0])
            break

    # ======================================================
    # 8. MARITAL STATUS & DEPENDENTS (Judgment / Petition)
    # ======================================================
    marital_patterns = {
        "married": ["married", "husband", "wife", "spouse"],
        "single": ["single", "unmarried", "bachelor", "spinster", "divorced"]
    }
    for status, keywords in marital_patterns.items():
        if any(kw in judgment_text for kw in keywords) or any(kw in petition_text for kw in keywords):
            suggestions["marital_status"] = status
            break

    dependents_patterns = [
        r'\b(\d{1,2})\s*dependents\b',
        r'\bno\.\s*of\s*dependents?\s*(?:is|:)?\s*(\d{1,2})\b',
        r'\bnumber\s*of\s*dependents?\s*(?:is|:)?\s*(\d{1,2})\b'
    ]
    for pattern in dependents_patterns:
        matches = re.findall(pattern, judgment_text)
        if not matches:
            matches = re.findall(pattern, petition_text)
        if not matches:
            matches = re.findall(pattern, full_text_lower)
        if matches:
            suggestions["dependents"] = int(matches[0])
            break

    # ======================================================
    # 9. NAMES (Judgment / Petition)
    # ======================================================
    name_patterns = [
        r'(?:name\s+of\s+)?(?:injured|deceased|claimant|victim)\s*(?:name)?\s*:\s*([A-Za-z\s\.]{3,30})',
        r'\bshri\b\s*([A-Za-z\s\.]{3,30})'
    ]
    for pattern in name_patterns:
        matches = re.findall(pattern, judgment_text)
        if not matches:
            matches = re.findall(pattern, petition_text)
        if not matches:
            matches = re.findall(pattern, full_text_lower)
        if matches:
            clean_name = re.sub(r'\s+', ' ', matches[0].strip()).title()
            if clean_name.endswith(' S'):
                clean_name = clean_name[:-2].strip()
            if not any(stop_word in clean_name.lower() for stop_word in ["father", "son", "daughter", "wife", "husband", "address"]):
                suggestions["name"] = clean_name
                break

    father_patterns = [
        r'(?:father|husband)\s*(?:s\s*)?name\s*:\s*([A-Za-z\s\.]{3,30})',
        r'\bs\/o\b\s*(?:shri)?\s*([A-Za-z\s\.]{3,30})',
        r'\bw\/o\b\s*(?:shri)?\s*([A-Za-z\s\.]{3,30})'
    ]
    for pattern in father_patterns:
        matches = re.findall(pattern, judgment_text)
        if not matches:
            matches = re.findall(pattern, petition_text)
        if not matches:
            matches = re.findall(pattern, full_text_lower)
        if matches:
            clean_father = re.sub(r'\s+', ' ', matches[0].strip()).title()
            suggestions["father_name"] = clean_father
            break

    return suggestions
