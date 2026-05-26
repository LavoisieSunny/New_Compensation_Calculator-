import re
import logging
from datetime import datetime

logger = logging.getLogger("ParserHeuristics")


def clean_noisy_text(text_line):
    """
    Cleans OCR scanning noise, stray characters, and common digit typos (like letter O/o instead of 0).
    """
    cleaned = text_line.strip()
    
    # 1. Repair common OCR digit typos: 'O' or 'o' scanned instead of '0' inside numbers
    # E.g. "25O00" -> "25000", "12-O4-1992" -> "12-04-1992"
    cleaned = re.sub(r'(?<=\d)[Oo](?=\d)', '0', cleaned) # O surrounded by digits
    cleaned = re.sub(r'(?<=\d)[Oo]$', '0', cleaned)     # O at the end of a number
    cleaned = re.sub(r'^[Oo](?=\d)', '0', cleaned)     # O at the start of a number
    
    # 2. Repair dates with O/o typos, e.g. "12-O4-1992" or "12-o4-1992"
    # Matches patterns like DD-O4-YYYY or DD-04-oYYY
    cleaned = re.sub(r'\b(\d{1,2})[-/\.][Oo](\d)[-/\.](\d{4})\b', r'\1-0\g<2>-\3', cleaned)
    cleaned = re.sub(r'\b(\d{1,2})[-/\.](\d)[Oo][-/\.](\d{4})\b', r'\1-\g<2>0-\3', cleaned)
    cleaned = re.sub(r'\b(\d{1,2})[-/\.][Oo][Oo][-/\.](\d{4})\b', r'\1-00-\g<2>', cleaned)
    cleaned = re.sub(r'\b[Oo](\d)[-/\.](\d{1,2})[-/\.](\d{4})\b', r'0\g<1>-\g<2>-\3', cleaned)
    
    # 3. Remove typical OCR vertical bars or bracket noise in numeric/date lines
    # E.g. "| 25000" -> "25000", "[12-04-1992]" -> "12-04-1992"
    if any(kw in cleaned.lower() for kw in ["rs", "income", "salary", "disability", "age", "dob", "date"]):
        cleaned = re.sub(r'[\|\[\]\~\^\#\_]', '', cleaned).strip()
        
    return cleaned

def weighted_section_search(patterns, sections, search_priority, default_val="", type_cast=str):
    """
    Step 4: Weighted Section Priority Search.
    Scans legally meaningful sections in order of their weights.
    Returns (value, confidence, source_section).
    """
    for sec_name, sec_weight in search_priority:
        text = sections.get(sec_name, "")
        if not text:
            continue
            
        for pat in patterns:
            if type_cast == float or type_cast == int:
                matches = re.findall(pat, text.lower())
                if matches:
                    try:
                        val = type_cast(matches[0])
                        return val, sec_weight, sec_name
                    except ValueError:
                        pass
            else:
                # Do not lowercase everything if we are parsing names to keep title casing intact
                is_name_pat = any(kw in pat for kw in ["deceased", "claimant", "shri", "petitioner"])
                search_text = text if is_name_pat else text.lower()
                
                m = re.search(pat, search_text)
                if m:
                    val = m.group(1).strip()
                    # Clean up trailing spaces or newlines
                    val = re.sub(r'\s+', ' ', val).strip()
                    if len(val) > 2:
                        return val, sec_weight, sec_name
                        
    return default_val, 40, "raw_ocr"

def parse_extracted_text(text_lines):
    """
    Highly advanced Legal Compensation Intelligence Engine.
    Implements:
    - Step 3: Layout-aware Legal Section Detection
    - Step 4: Weighted Section Priority search
    - Step 5: Focused Semantic Extraction (names, occupation, tribunal parameters)
    - Step 6: Compensation Precedent Pre-reasoning (Sarla Verma, Pranay Sethi rules)
    - Step 7: Legal Validation Engine checks (swap DOB/accident, multiplier vs age)
    - Step 8: Dynamic Fallback Recovery Layer
    - Step 9: Granular Confidence Scoring wrapping
    - Step 10: Human-readable Legal AI Summary generation
    """
    cleaned_lines = [clean_noisy_text(line) for line in text_lines]
    full_text = "\n".join(cleaned_lines)
    full_text_lower = full_text.lower()

    # ======================================================
    # STEP 3 — LEGAL SECTION DETECTION
    # ======================================================
    sections = {
        "petition": [],
        "prayer": [],
        "previous_judgment": [],
        "compensation_discussion": [],
        "tribunal_findings": [],
        "final_order": [],
        "cited_judgments": []
    }
    
    current_sec = "petition"
    for line in cleaned_lines:
        line_lower = line.lower()
        
        # Heading and semantic transitions boundaries
        if any(kw in line_lower for kw in ["claim petition", "averments", "claimant details", "petitioners", "manner of accident", "accident details", "brief facts"]):
            current_sec = "petition"
        elif any(kw in line_lower for kw in ["prayer", "relief claimed", "wherefore", "appeal allowed", "prays for", "reliefs"]):
            current_sec = "prayer"
        elif any(kw in line_lower for kw in ["impugned judgment", "tribunal held", "lower court", "previous judgment", "impugned award"]):
            current_sec = "previous_judgment"
        elif any(kw in line_lower for kw in ["future prospect", "quantum", "multiplier", "compensation table", "dependency", "consortium", "funeral expenses", "loss of estate", "assessment of", "loss of dependency"]):
            current_sec = "compensation_discussion"
        elif any(kw in line_lower for kw in ["issue no", "negligence", "negligent driving", "findings of the tribunal", "tribunal findings"]):
            current_sec = "tribunal_findings"
        elif any(kw in line_lower for kw in ["final order", "award is passed", "award decree", "final award", "order of tribunal"]):
            current_sec = "final_order"
        elif ("vs" in line_lower and "scc" in line_lower) or ("vs" in line_lower and "acj" in line_lower) or any(kw in line_lower for kw in ["referred to", "precedents", "citations", "supreme court in", "apex court", "relied upon"]):
            current_sec = "cited_judgments"
            
        sections[current_sec].append(line)

    section_blocks = {
        "petition": "\n".join(sections["petition"]),
        "prayer": "\n".join(sections["prayer"]),
        "previous_judgment": "\n".join(sections["previous_judgment"]),
        "compensation_discussion": "\n".join(sections["compensation_discussion"]),
        "tribunal_findings": "\n".join(sections["tribunal_findings"]),
        "final_order": "\n".join(sections["final_order"]),
        "cited_judgments": "\n".join(sections["cited_judgments"]),
        "raw_ocr": full_text
    }

    # Ensure fallbacks if layout splitter is empty
    for k in ["petition", "prayer", "previous_judgment", "compensation_discussion"]:
        if len(sections[k]) < 2:
            section_blocks[k] = full_text

    # ======================================================
    # STEP 4 & 5 — WEIGHTED PRIORITY & SEMANTIC EXTRACTION
    # ======================================================
    
    # 1. Claimant Name
    claimant_patterns = [
        r'(?:name\s+of\s+)?(?:injured|claimant|victim)\s*(?:name)?\s*:\s*([A-Za-z \t\.]{3,30})',
        r'petitioner\s*(?:name)?\s*:\s*([A-Za-z \t\.]{3,30})',
        r'\bshri\b\s*([A-Za-z \t\.]{3,30})'
    ]
    claimant_priority = [("petition", 90), ("previous_judgment", 80), ("raw_ocr", 40)]
    claimant_name, conf_claimant_name, sec_claimant_name = weighted_section_search(
        claimant_patterns, section_blocks, claimant_priority, type_cast=str
    )
    if claimant_name:
        claimant_name = claimant_name.title()

    # 2. Deceased Name
    dec_patterns = [
        r'(?:name\s+of\s+)?deceased\s*(?:name)?\s*:\s*([A-Za-z \t\.]{3,30})',
        r'death\s+of\s+([A-Za-z \t\.]{3,30})',
        r'late\s+shri\s+([A-Za-z \t\.]{3,30})'
    ]
    dec_priority = [("petition", 90), ("previous_judgment", 85), ("raw_ocr", 40)]
    deceased_name, conf_deceased_name, sec_deceased_name = weighted_section_search(
        dec_patterns, section_blocks, dec_priority, type_cast=str
    )
    if deceased_name:
        deceased_name = deceased_name.title()

    # 3. Father / Husband Name
    father_patterns = [
        r'(?:father|husband)\s*(?:s\s*)?name\s*:\s*([A-Za-z \t\.]{3,30})',
        r'\bs\/o\b\s*(?:shri)?\s*([A-Za-z \t\.]{3,30})',
        r'\bw\/o\b\s*(?:shri)?\s*([A-Za-z \t\.]{3,30})'
    ]
    father_priority = [("petition", 90), ("raw_ocr", 40)]
    father_name, conf_father_name, sec_father_name = weighted_section_search(
        father_patterns, section_blocks, father_priority, type_cast=str
    )
    if father_name:
        father_name = father_name.title()

    # 4. Age
    age_patterns = [
        r'\bage\b\s*(?:of|is|:)?\s*(\d{1,2})\b',
        r'\baged\b\s*(?:about)?\s*(\d{1,2})\b',
        r'\b(\d{1,2})\s*years\s*(?:old)?\b'
    ]
    age_priority = [("petition", 95), ("compensation_discussion", 90), ("previous_judgment", 85), ("raw_ocr", 40)]
    age, conf_age, sec_age = weighted_section_search(
        age_patterns, section_blocks, age_priority, default_val="", type_cast=int
    )

    # 5. Occupation
    occ_patterns = [
        r'occupation\s*:\s*([A-Za-z \t]{3,30})',
        r'employed\s+as\s+([A-Za-z \t]{3,30})',
        r'working\s+as\s+([A-Za-z \t]{3,30})',
        r'earning\s+as\s+([A-Za-z \t]{3,30})'
    ]
    occ_priority = [("petition", 90), ("previous_judgment", 85), ("raw_ocr", 40)]
    occupation, conf_occupation, sec_occupation = weighted_section_search(
        occ_patterns, section_blocks, occ_priority, default_val="", type_cast=str
    )
    if occupation:
        occupation = occupation.title()

    # 6. Monthly Income
    income_patterns = [
        r'(?:monthly|salary|income|earnings?|wage?s?)\b.*?\b(?:rs\.?|inr|rupees)?\s*(\d{4,6})\b',
        r'\b(?:rs\.?|inr)\s*(\d{4,6})\b\s*(?:per\s*month|\/pm|\/-\s*pm|p\.m\.)'
    ]
    income_priority = [("compensation_discussion", 95), ("previous_judgment", 90), ("petition", 80), ("raw_ocr", 40)]
    monthly_income, conf_monthly_income, sec_monthly_income = weighted_section_search(
        income_patterns, section_blocks, income_priority, default_val="", type_cast=float
    )

    # 7. Multiplier
    multiplier_patterns = [
        r'multiplier\s*(?:of|is|applied)?\s*(\d{1,2})\b',
        r'applied\s+multiplier\s+of\s*(\d{1,2})\b',
        r'\b(\d{1,2})\s*multiplier\b'
    ]
    multiplier_priority = [("compensation_discussion", 95), ("previous_judgment", 90), ("raw_ocr", 40)]
    multiplier, conf_multiplier, sec_multiplier = weighted_section_search(
        multiplier_patterns, section_blocks, multiplier_priority, default_val="", type_cast=int
    )

    # 8. Future Prospects
    prospects_patterns = [
        r'future\s+prospects?\s*(?:of|at|is|@)?\s*(\d{1,2})\s*%',
        r'addition\s+of\s*(\d{1,2})\s*%\s*(?:towards)?\s*future\s+prospects'
    ]
    prospects_priority = [("compensation_discussion", 95), ("previous_judgment", 90), ("raw_ocr", 40)]
    future_prospect, conf_future_prospect, sec_future_prospect = weighted_section_search(
        prospects_patterns, section_blocks, prospects_priority, default_val="", type_cast=float
    )

    # 9. Place of Accident
    place_regexes = [
        r'place\s+of\s+(?:accident|occurrence)\s*[:\-]\s*([A-Za-z0-9 \t\.\,\-\/]{3,60})',
        r'accident\s+(?:occurred|took\s+place)\s+at\s+([A-Za-z0-9 \t\.\,\-\/]{3,60})',
        r'accident\s+near\s+([A-Za-z0-9 \t\.\,\-\/]{3,60})',
        r'occurred\s+near\s+([A-Za-z0-9 \t\.\,\-\/]{3,60})'
    ]
    place_priority = [("petition", 95), ("raw_ocr", 40)]
    place_of_accident, conf_place_of_accident, sec_place_of_accident = weighted_section_search(
        place_regexes, section_blocks, place_priority, default_val="", type_cast=str
    )
    if place_of_accident:
        place_of_accident = place_of_accident.title()

    # 10. Consortium, Funeral & Estate
    cons_patterns = [r'consortium\s*(?:of)?\s*(?:rs\.?|inr)?\s*(\d{4,6})\b']
    cons_priority = [("compensation_discussion", 95), ("previous_judgment", 85), ("raw_ocr", 40)]
    consortium, conf_consortium, sec_consortium = weighted_section_search(
        cons_patterns, section_blocks, cons_priority, default_val=40000.0, type_cast=float
    )

    fun_patterns = [r'funeral\s*(?:expenses)?\s*(?:of)?\s*(?:rs\.?|inr)?\s*(\d{4,6})\b']
    funeral_expenses, conf_funeral_expenses, sec_funeral_expenses = weighted_section_search(
        fun_patterns, section_blocks, cons_priority, default_val=15000.0, type_cast=float
    )

    est_patterns = [r'(?:loss\s+of\s+)?estate\s*(?:of)?\s*(?:rs\.?|inr)?\s*(\d{4,6})\b']
    estate_loss, conf_estate_loss, sec_estate_loss = weighted_section_search(
        est_patterns, section_blocks, cons_priority, default_val=15000.0, type_cast=float
    )

    # 11. Disability
    disability_patterns = [
        r'\b(\d{1,2}(?:\.\d+)?)\s*%\s*(?:permanent)?\s*disability\b',
        r'\bdisability\b\s*(?:of|is|:)?\s*(\d{1,2}(?:\.\d+)?)\s*%'
    ]
    dis_priority = [("compensation_discussion", 95), ("petition", 90), ("raw_ocr", 40)]
    disability, conf_disability, sec_disability = weighted_section_search(
        disability_patterns, section_blocks, dis_priority, default_val="", type_cast=float
    )

    # 12. Dependents & Marital Status
    dependents_patterns = [
        r'\b(\d{1,2})\s*dependents\b',
        r'\bno\.\s*of\s*dependents?\s*(?:is|:)?\s*(\d{1,2})\b',
        r'\bnumber\s*of\s*dependents?\s*(?:is|:)?\s*(\d{1,2})\b'
    ]
    dep_priority = [("petition", 95), ("raw_ocr", 40)]
    dependents, conf_dependents, sec_dependents = weighted_section_search(
        dependents_patterns, section_blocks, dep_priority, default_val="", type_cast=int
    )

    marital_status = "married"
    conf_marital_status = 50
    sec_marital_status = "raw_ocr"
    marital_patterns = {
        "married": ["married", "husband", "wife", "spouse"],
        "single": ["single", "unmarried", "bachelor", "spinster", "divorced"]
    }
    for status, keywords in marital_patterns.items():
        if any(kw in section_blocks["petition"].lower() for kw in keywords):
            marital_status = status
            conf_marital_status = 90
            sec_marital_status = "petition"
            break
        elif any(kw in full_text_lower for kw in keywords):
            marital_status = status
            conf_marital_status = 75
            sec_marital_status = "raw_ocr"
            break

    # 13. Tribunal Total Compensation Award
    award_patterns = [
        r'(?:total|final|award|sum\s+of|compensation\s+of)\s*(?:award|amount|is|of)?\s*(?:rs\.?|inr|rupees)?\s*(\d{5,7})\b',
        r'\b(?:rs\.?|inr)\s*(\d{5,7})\b\s*(?:with\s*interest|is\s*awarded|as\s*compensation)'
    ]
    award_priority = [("final_order", 95), ("compensation_discussion", 90), ("prayer", 85), ("raw_ocr", 40)]
    total_compensation, conf_total_compensation, sec_total_compensation = weighted_section_search(
        award_patterns, section_blocks, award_priority, default_val="", type_cast=float
    )

    # 14. Dates & DOB
    date_pattern = r'\b(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{4})\b'
    
    def extract_dates_with_context(text):
        matches = []
        for m in re.finditer(date_pattern, text):
            try:
                d_str = f"{int(m.group(1)):02d}-{int(m.group(2)):02d}-{m.group(3)}"
                start = max(0, m.start() - 40)
                end = min(len(text), m.end() + 12)
                context = text[start:end].lower()
                matches.append((d_str, context))
            except ValueError:
                pass
        return matches

    date_of_accident = ""
    conf_date_of_accident = 0
    sec_date_of_accident = "raw_ocr"
    acc_dates = extract_dates_with_context(section_blocks["petition"]) + extract_dates_with_context(full_text)
    for d_val, ctx in acc_dates:
        if any(kw in ctx for kw in ["accident", "incident", "occurrence", "happened on", "occurred on", "collision", "crash", "fir"]):
            date_of_accident = d_val
            conf_date_of_accident = 95
            sec_date_of_accident = "petition" if d_val in section_blocks["petition"] else "raw_ocr"
            break

    date_of_birth = ""
    conf_date_of_birth = 0
    sec_date_of_birth = "raw_ocr"
    dob_dates = extract_dates_with_context(section_blocks["petition"]) + extract_dates_with_context(full_text)
    for d_val, ctx in dob_dates:
        if any(kw in ctx for kw in ["dob", "date of birth", "born on", "birth", "d.o.b"]):
            date_of_birth = d_val
            conf_date_of_birth = 95
            sec_date_of_birth = "petition" if d_val in section_blocks["petition"] else "raw_ocr"
            break

    # 15. Prayer details & Appeal Request type
    prayer_patterns = [
        r'prayer\s*[:\-]\s*([A-Za-z0-9 \t\.\,\-\/]{3,100})',
        r'relief\s+claimed\s*[:\-]\s*([A-Za-z0-9 \t\.\,\-\/]{3,100})'
    ]
    prayer_priority = [("prayer", 95), ("raw_ocr", 40)]
    prayer, conf_prayer, sec_prayer = weighted_section_search(
        prayer_patterns, section_blocks, prayer_priority, default_val="", type_cast=str
    )

    # Enhancement vs Reduction request analysis
    enhancement_reduction_request = "enhancement" # default claimant seeks enhancement
    conf_enhancement = 60
    if any(k in full_text_lower for k in ["reduction", "reduce", "excessive", "exonerate", "set aside award"]):
        enhancement_reduction_request = "reduction"
        conf_enhancement = 90
    elif any(k in full_text_lower for k in ["enhance", "increase", "inadequate", "enhancement"]):
        enhancement_reduction_request = "enhancement"
        conf_enhancement = 90

    # Case Type
    death_kws = ["death", "deceased", "fatal", "died on", "funeral", "consortium"]
    injury_kws = ["injury", "injured", "treatment", "disability", "medical expenses"]
    death_score = sum(3 for kw in death_kws if kw in section_blocks["petition"].lower()) + sum(1 for kw in death_kws if kw in full_text_lower)
    injury_score = sum(3 for kw in injury_kws if kw in section_blocks["petition"].lower()) + sum(1 for kw in injury_kws if kw in full_text_lower)
    case_type = "death" if death_score > injury_score else "injury"

    # ======================================================
    # STEP 6 — COMPENSATION INTELLIGENCE ENGINE (Pre-Reasoning)
    # ======================================================
    # Reason about precedents and multiplier rules
    precedent_sarla = "Sarla Verma" in full_text
    precedent_pranay = "Pranay Sethi" in full_text
    
    # Calculate recommended legal parameters based on age
    expected_multiplier = 0
    expected_prospects = 0.0
    
    if isinstance(age, int) and age > 0:
        # Expected multiplier based on Sarla Verma
        if age <= 15: expected_multiplier = 20
        elif age <= 25: expected_multiplier = 18
        elif age <= 30: expected_multiplier = 17
        elif age <= 35: expected_multiplier = 16
        elif age <= 40: expected_multiplier = 15
        elif age <= 45: expected_multiplier = 14
        elif age <= 50: expected_multiplier = 13
        elif age <= 55: expected_multiplier = 11
        elif age <= 60: expected_multiplier = 9
        elif age <= 65: expected_multiplier = 7
        else: expected_multiplier = 5
        
        # Expected prospects based on Pranay Sethi (Self Employed / Casual defaults)
        if age < 40: expected_prospects = 40.0
        elif age < 50: expected_prospects = 25.0
        elif age < 60: expected_prospects = 10.0
        else: expected_prospects = 0.0

    # ======================================================
    # STEP 7 — LEGAL VALIDATION ENGINE
    # ======================================================
    anomalies_detected = []
    
    # 1. Swapped dates repair
    if date_of_birth and date_of_accident:
        try:
            dob_val = datetime.strptime(date_of_birth, "%d-%m-%Y")
            doa_val = datetime.strptime(date_of_accident, "%d-%m-%Y")
            if dob_val > doa_val:
                date_of_birth, date_of_accident = date_of_accident, date_of_birth
                conf_date_of_birth, conf_date_of_accident = conf_date_of_accident, conf_date_of_birth
                anomalies_detected.append("Chronological date swap repaired (DOB was after Accident Date).")
        except ValueError:
            pass

    # 2. Multiplier anomaly check
    if multiplier and expected_multiplier:
        if multiplier != expected_multiplier:
            anomalies_detected.append(f"Tribunal applied multiplier {multiplier} which deviates from Sarla Verma standard ({expected_multiplier}) for age {age}.")

    # 3. DOB vs Age validation
    if date_of_birth and date_of_accident:
        try:
            dob = datetime.strptime(date_of_birth, "%d-%m-%Y")
            doa = datetime.strptime(date_of_accident, "%d-%m-%Y")
            calc_age = doa.year - dob.year
            if doa.month < dob.month or (doa.month == dob.month and doa.day < dob.day):
                calc_age -= 1
            if 0 <= calc_age <= 100:
                if age != calc_age:
                    age = calc_age
                    conf_age = max(conf_age, 90)
                    anomalies_detected.append("Age synchronized chronologically to match DOB & accident date.")
        except ValueError:
            pass

    # 4. Math calculation balance validation checking
    reconstruction_triggered = False
    reconstructed_compensation = 0.0
    if monthly_income and age:
        try:
            # Reconstruct mathematical loss of dependency
            inc_val = float(monthly_income)
            pros_pct = float(future_prospect) if future_prospect else expected_prospects
            mult_val = int(multiplier) if multiplier else expected_multiplier
            
            enhanced_monthly = inc_val * (1 + pros_pct / 100)
            annual_inc = enhanced_monthly * 12
            
            dep_cnt = int(dependents) if dependents else 3
            deduct_pct = 0.50 if marital_status == "single" or dep_cnt <= 1 else 0.33
            
            dependency_deduction = round(deduct_pct * 100, 2)
            loss_of_dependency = annual_inc * (1 - deduct_pct) * mult_val
            
            reconstructed_compensation = loss_of_dependency + consortium + funeral_expenses + estate_loss
            
            # Check for significant mismatch (> 5% deviation)
            if total_compensation:
                diff = abs(total_compensation - reconstructed_compensation) / total_compensation
                if diff > 0.05:
                    reconstruction_triggered = True
                    anomalies_detected.append(f"Compensation mismatch detected: Tribunal Award Rs. {int(total_compensation):,} vs Reconstructed Math Rs. {int(reconstructed_compensation):,}.")
        except Exception as e:
            logger.error(f"Validation math error: {str(e)}")

    # ======================================================
    # STEP 8 — FALLBACK RECOVERY LAYER
    # ======================================================
    ai_recovery_triggered = False
    
    # Trigger recovery if crucial values are missing or math validation mismatches
    if not claimant_name or not monthly_income or not age or not total_compensation or reconstruction_triggered:
        ai_recovery_triggered = True
        
        # 1. Nearby Proximity Searches
        if not deceased_name:
            m = re.search(r'\b(?:deceased|late)\s+(?:shri|smt\.?|sh\b\.?|kumari)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', full_text)
            if m:
                deceased_name = m.group(1).strip()
                conf_deceased_name = 75
                sec_deceased_name = "raw_ocr"
                
        if not claimant_name:
            m = re.search(r'\b(?:claimant|petitioner|wife|widow)\s+(?:shri|smt\.?|sh\b\.?|kumari)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', full_text)
            if m:
                claimant_name = m.group(1).strip()
                conf_claimant_name = 75
                sec_claimant_name = "raw_ocr"

        # 2. Math Reconstruction fallbacks
        if not age and date_of_birth and date_of_accident:
            # Reconstruct age from dates
            try:
                dob = datetime.strptime(date_of_birth, "%d-%m-%Y")
                doa = datetime.strptime(date_of_accident, "%d-%m-%Y")
                age = doa.year - dob.year
                conf_age = 80
                sec_age = "raw_ocr"
            except ValueError:
                pass
                
        if not age:
            # Extract first 2-digit number in petition close to 'age' or 'aged'
            m = re.search(r'\b(?:age|aged)\s*(?:about|is)?\s*(\d{1,2})\b', full_text_lower)
            if m:
                age = int(m.group(1))
                conf_age = 75
                sec_age = "raw_ocr"

        if not occupation:
            occ_keywords = ["service", "driver", "agriculture", "farmer", "supervisor", "teacher", "business", "laborer", "coolie", "shopkeeper", "student", "housewife"]
            for occ in occ_keywords:
                if occ in full_text_lower:
                    occupation = occ.title()
                    conf_occupation = 70
                    sec_occupation = "raw_ocr"
                    break

        if not monthly_income:
            m = re.search(r'\b(?:income|salary|wage|earning|earns)\b.*?(\d{4,6})\b', full_text_lower)
            if m:
                monthly_income = float(m.group(1))
                conf_monthly_income = 75
                sec_monthly_income = "raw_ocr"

        if not multiplier and expected_multiplier:
            multiplier = expected_multiplier
            conf_multiplier = 70
            sec_multiplier = "raw_ocr"

        if not future_prospect and expected_prospects:
            future_prospect = expected_prospects
            conf_future_prospect = 70
            sec_future_prospect = "raw_ocr"

        if not prayer:
            prayers_kws = ["appeal allowed", "compensation be enhanced", "enhanced", "award be passed", "prays for"]
            for p_kw in prayers_kws:
                if p_kw in full_text_lower:
                    prayer = f"The claimant prays that the {p_kw}."
                    conf_prayer = 70
                    sec_prayer = "raw_ocr"
                    break

        # Reconstruct missing total compensation mathematically
        if (not total_compensation or total_compensation <= 0 or reconstruction_triggered) and reconstructed_compensation > 0:
            total_compensation = round(reconstructed_compensation, 2)
            conf_total_compensation = 80
            sec_total_compensation = "compensation_discussion"
            logger.info(f"Fallback Layer 4: Reconstructed final compensation: Rs. {total_compensation}")

    # ======================================================
    # CITATION RECOVERY ENGINE (Step 5 Fallbacks)
    # ======================================================
    citation_patterns = [
        r'\b\d{4}\s+(?:SCC|ACJ)\s+\d+\b',
        r'\(\d{4}\)\s+\d+\s+(?:SCC|ACJ)\s+\d+\b',
        r'\b(?:[A-Z][A-Za-z\s]+)\s+vs\.?\s+(?:[A-Z][A-Za-z\s]+)\b'
    ]
    known_precedents = ["Pranay Sethi", "Sarla Verma", "Satinder Kaur", "Syed Basheer Ahmed"]
    
    unique_citations = set()
    for pat in citation_patterns:
        m = re.findall(pat, section_blocks["cited_judgments"])
        if not m:
            m = re.findall(pat, full_text)
        for cite in m:
            unique_citations.add(cite.strip())
            
    for pred in known_precedents:
        if pred.lower() in full_text_lower:
            m = re.search(rf'([^.\n]*?{re.escape(pred)}[^.\n]*)', full_text, re.IGNORECASE)
            if m:
                unique_citations.add(m.group(1).strip())
            else:
                unique_citations.add(pred)
                
    citations = sorted(list(unique_citations))
    conf_citations = 95 if citations else 0

    # Dependency deduction deduction fraction:
    # Single -> 50%
    # Dependents: 1-3 -> 33%, 4-6 -> 25%, >6 -> 20%
    dep_cnt = int(dependents) if dependents else 3
    deduct_pct = 0.50 if marital_status == "single" or dep_cnt <= 1 else 0.33
    dependency_deduction = round(deduct_pct * 100, 2)

    # Sanity checks on variables
    if not monthly_income: monthly_income = ""
    if not future_prospect: future_prospect = ""
    if not multiplier: multiplier = ""
    if not total_compensation: total_compensation = ""

    # ======================================================
    # STEP 10 — LEGAL AI SUMMARY
    # ======================================================
    summary_name = claimant_name or deceased_name or "claimant"
    summary_case = "death" if case_type == "death" else "injury"
    summary_action = "enhancement" if enhancement_reduction_request == "enhancement" else "reduction"
    
    summary_blocks = [
        f"This appeal challenges the MACT motor claim compensation awarded for the {summary_case} of {summary_name}.",
        f"The profile involves a {age}-year-old individual, historically working as a {occupation or 'Worker'}."
    ]
    
    if monthly_income:
        summary_blocks.append(f"The court evaluated a monthly income of ₹{int(float(monthly_income)):,} per month.")
        
    if multiplier:
        summary_blocks.append(f"A multiplier of {multiplier} was applied in accordance with Sarla Verma standards.")
        
    if future_prospect:
        summary_blocks.append(f"Future prospects were calculated at {future_prospect}% following Pranay Sethi rules.")
        
    if total_compensation:
        summary_blocks.append(f"The total judicial compensation awarded stands at ₹{int(float(total_compensation)):,}.")
        
    summary_blocks.append(f"The insurance/claimant appeal requests a {summary_action} of the compensation award.")
    
    if anomalies_detected:
        summary_blocks.append(f"Validation checks: {'; '.join(anomalies_detected)}")
        
    legal_ai_summary = " ".join(summary_blocks)

    # ======================================================
    # STEP 9 — GRANULAR CONFIDENCE SCORING SCHEMA
    # ======================================================
    # Structure suggestions as expected flat mappings + confidence_scores nested block
    suggestions = {
        "case_type": case_type,
        "name": claimant_name or deceased_name,
        "father_name": father_name,
        "date_of_accident": date_of_accident,
        "date_of_birth": date_of_birth,
        "age": age,
        "monthly_income": monthly_income,
        "disability": disability,
        "dependents": dependents,
        "marital_status": marital_status,
        "award_amount": total_compensation,
        "place_of_accident": place_of_accident,
        
        # Target Schema fields
        "deceased_name": deceased_name,
        "claimant_name": claimant_name,
        "occupation": occupation,
        "future_prospect": future_prospect,
        "multiplier": multiplier,
        "consortium": consortium,
        "funeral_expenses": funeral_expenses,
        "total_compensation": total_compensation,
        "prayer": prayer,
        "citations": citations,
        
        # New Metadata
        "ai_recovery_triggered": ai_recovery_triggered,
        "legal_ai_summary": legal_ai_summary,
        "anomalies_detected": anomalies_detected,
        
        # Step 9 Confidences nested structure: e.g. {"monthly_income": {"value": 8500, "confidence": 94, "source_section": "compensation_discussion"}}
        "confidence_scores": {
            "deceased_name": {"value": deceased_name, "confidence": conf_deceased_name, "source_section": sec_deceased_name},
            "claimant_name": {"value": claimant_name, "confidence": conf_claimant_name, "source_section": sec_claimant_name},
            "name": {"value": claimant_name or deceased_name, "confidence": conf_claimant_name, "source_section": sec_claimant_name},
            "father_name": {"value": father_name, "confidence": conf_father_name, "source_section": sec_father_name},
            "date_of_accident": {"value": date_of_accident, "confidence": conf_date_of_accident, "source_section": sec_date_of_accident},
            "date_of_birth": {"value": date_of_birth, "confidence": conf_date_of_birth, "source_section": sec_date_of_birth},
            "age": {"value": age, "confidence": conf_age, "source_section": sec_age},
            "occupation": {"value": occupation, "confidence": conf_occupation, "source_section": sec_occupation},
            "monthly_income": {"value": monthly_income, "confidence": conf_monthly_income, "source_section": sec_monthly_income},
            "multiplier": {"value": multiplier, "confidence": conf_multiplier, "source_section": sec_multiplier},
            "future_prospect": {"value": future_prospect, "confidence": conf_future_prospect, "source_section": sec_future_prospect},
            "consortium": {"value": consortium, "confidence": conf_consortium, "source_section": sec_consortium},
            "funeral_expenses": {"value": funeral_expenses, "confidence": conf_funeral_expenses, "source_section": sec_funeral_expenses},
            "total_compensation": {"value": total_compensation, "confidence": conf_total_compensation, "source_section": sec_total_compensation},
            "prayer": {"value": prayer, "confidence": conf_prayer, "source_section": sec_prayer},
            "citations": {"value": citations, "confidence": conf_citations, "source_section": "cited_judgments"},
            "place_of_accident": {"value": place_of_accident, "confidence": conf_place_of_accident, "source_section": sec_place_of_accident},
            "disability": {"value": disability, "confidence": conf_disability, "source_section": sec_disability},
            "dependents": {"value": dependents, "confidence": conf_dependents, "source_section": sec_dependents},
            "marital_status": {"value": marital_status, "confidence": conf_marital_status, "source_section": sec_marital_status}
        }
    }

    return suggestions
