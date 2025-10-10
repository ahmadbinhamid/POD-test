SIGNATURE_PROMPT = """
Analyze the image carefully to identify handwritten signatures and generate JSON output.

**Key: `receiver_signature`**
- Look for handwritten signatures in the "RECEIVING STAMP AREA" or "Notation Area (Customer Use ONLY)"
- Check fields labeled: "SIGNATURE", "REC'D BY", "RECEIVED BY", "PRINT NAME"
- **Important**: Only handwritten text/signatures count as "yes"
- If only a printed name, date, or stamp is present without handwritten signature: "no"
- If field is empty or only contains printed text: "no"

**Values:**
- `"yes"`: Handwritten signature is clearly present
- `"no"`: No handwritten signature (only dates, printed names, or stamps)

**Output Format:**
```json
{
  "receiver_signature": "yes" | "no"
}
```

First provide reasoning explaining what you observe in signature fields, then generate JSON output.
"""

BILL_OF_LADING_PROMPT = """
Extract bill number and date from the Bill of Lading document.

**Key Extraction Rules:**

1. **`bill_no`**: 
   - Look for "Bill of Lading:" label followed by a number (usually 8-9 digits)
   - Extract ONLY the numeric value after "Bill of Lading:" 
   - Example: "Bill of Lading: 17540697" → extract "17540697"

2. **`date`**:
   - Look for dates in the "RECEIVING STAMP AREA" or "Notation Area (Customer Use ONLY)" only
   - Common locations: must be inside the "RECEIVING STAMP AREA", or "Notation Area (Customer Use ONLY)" area
   - Extract in the exact format shown (MM/DD/YYYY, MM-DD-YYYY, etc.)
   - **Do NOT extract**: time stamps, appointment dates, or delivery dates
   - If date is partially obscured or unclear: extract visible portion only
   - If completely unreadable: "null"

**Output Format:**
```json
{
  "bill_no": "<string>",
  "date": "<string>" | "null"
}
```

First analyze the document identifying the bill number and date locations, then generate JSON output.
"""

CUSTOMER_ORDER_INFO_PROMPT = """
Extract the single total order quantity from the document WITHOUT any calculation or multiplication.

**Key: `total_order_quantity`**

**CRITICAL RULE: DO NOT PERFORM ANY MATH OPERATIONS**
- Find the pre-calculated total quantity value ONLY
- Do NOT sum individual line items
- Do NOT multiply any values
- Do NOT add quantities together

**Extraction Priority:**
1. **Primary**: Look for "TOTAL QTY", "TOTAL QUANTITY", or similar total fields
2. **Secondary**: If no total field, look for a single "QTY" value (not multiple rows)
3. **Last resort**: If multiple line items exist, extract the FIRST quantity value only

**Target Locations:**
- Summary/total rows in "CUSTOMER ORDER INFORMATION" section
- Pre-calculated total fields
- Single quantity entries

**Avoid These Actions:**
- Do NOT sum multiple rows: If you see QTY: 34, QTY: 35 → extract 34 (first value only)
- Do NOT multiply: If you see 2 x 34 → extract 34 (not 68)
- Do NOT add columns: QTY + HANDLING QTY → use QTY only

**Output Format:**
```json
{
  "total_order_quantity": <integer>
}
```

Find the single total quantity value without performing any mathematical operations, then generate JSON output.
"""

STAMP_LATEST_PROMPT = """
**Advanced Bill of Lading Stamp Analysis**

Analyze the "RECEIVING STAMP AREA", "Notation Area (Customer Use ONLY)" section meticulously. Extract information using these precise rules:

### Field Extraction Guidelines:

1. **stamp_exist**: 
   - **SEARCH CRITERIA**: Look for "RECEIVING STAMP AREA" or "Notation Area (Customer Use ONLY)" section or stamp impressions 
   - **STAMP INDICATORS**: Rectangular/circular stamp marks, company logos, official seals 
   - **VALUES**: "yes" (stamp area/impression found) | "no" (no stamp area) 

2. **seal_intact**: 
   - **KEYWORD TARGETS**: "Seal(s) Intact Y/N", "SEAL INTACT Y N", "Seal Intact? Y N" 
   - **CRITICAL RULE**: ONLY process if you find these EXACT field labels
   - **DETECTION METHOD**: Look for Y/N options, checkboxes, circles 
   - **MARKED INDICATORS**: 
     - "Y" circled/checked/marked = "yes" 
     - "N" circled/checked/marked = "no" 
     - Field present but unmarked = "empty" 
   - **IMPORTANT**: If you don't find these EXACT field labels, return "null"
   - **VALUES**: "yes" | "no" | "empty" | "null"

3. **pod_date**:
   - **KEYWORD TARGETS**: "DATE", "Date", "Signed Date", "Pick Up Appointment Date & Time", "Driver Arrival Time"
   - **LOCATION**: Must be inside "RECEIVING STAMP AREA" or "Notation Area (Customer Use ONLY)" (ignore header dates)
   - **POSITION ASSOCIATION**: Accept handwritten/printed values **to the right OR directly below** the keyword
   - **FORMAT PRESERVATION**: Extract exactly as handwritten/printed
   - **DATE REGEX**: Match short and long formats: `\d{1,2}[-/]\d{1,2}[-/]\d{2,4}`
   - **VALUES**: "<date_string>" | "empty" | "null"

4. **pod_sign** (Signature Detection):
   - **KEYWORD TARGETS**: "Signature", "PRINT", "RECEIVED BY", "Receiver Signature", "Rec'd by", etc.
   - **SIGNATURE CRITERIA**:
     - Any handwritten scribbles, cursive, initials, or full names written by hand
     - Accept as `"yes"` even if OCR normalizes cursive into letters
   - **EXCLUSIONS**: Only reject if the field is blank or filled with a printed/stamped label (not handwriting)
   - **VALUES**: "yes" (handwritten content) | "no" (printed only) | "empty" (field present but blank) | "null" (field absent)

5. **total_received**:
   **STEP 1 - FIND CARTON LABELS ONLY:**
   Look for these exact words/phrases (case-insensitive):
   - "CARTONS" 
   - "TOT CS REC"
   - "TOTAL CARTONS"
   - "Ctns Received" 
   - "Ctns. Received"

   **STEP 2 - GET NUMBER NEXT TO LABEL:**
   - Extract the number immediately after/below the carton label
   - Must be same line or directly below in same box

   **STEP 3 - IGNORE THESE COMPLETELY:**
   - Any number after "PALLET", "PLT", "PLTS" 
   - Any number after "APPT", "APPOINTMENT", "Appointment #", "APPOINTMENT #"
   - Any number after "TRAILER", "TRL", "UNIT" "TRAILER #'S", "Trailer #", "TRAILER #"
   - Any number after "SEAL #'S", "Seal Number", "seal"
   - Any number after "BDC", "DC" followed by 4 digits
   - Any 4-digit number that looks like time (1504, 1430, etc.)
   - Any standalone 4-digit number without a carton label

   **STEP 4 - RETURN VALUES:**
   - If carton label found with number → return the number
   - If carton label found but no number → "empty"  
   - If no carton label found → "null"

   **CRITICAL RULE:** 
   Only extract numbers that are directly associated with carton/cases labels("CARTONS", "TOT CS REC", "TOTAL CARTONS", "Ctns Received", "Ctns. Received") 
   If there's any doubt about what the number represents, return "null".

6. **damage**:
   - **KEYWORD TARGETS**: "DAMAGED", "Damage Kept", "D", "Damage", "Cartons Damaged", "Ctns Damaged", "Ctns. Damaged" 
   - **EXTRACTION RULE**: Look for NUMBERS in damage fields 
   - **VALIDATION**: Must be integer or zero, not letters 
   - **VALUES**: <integer> | "empty" | "null" 
 
7. **short**: 
   - **KEYWORD TARGETS**: "SHORT", "S", "Short", "S", "Cartons Short", "Ctns Short", "Ctns. Short" 
   - **EXTRACTION RULE**: Look for NUMBERS in short fields 
   - **VALUES**: <integer> | "empty" | "null" 
 
8. **over**: 
   - **KEYWORD TARGETS**: "OVER", "O", "Over", "O" 
   - **EXTRACTION RULE**: Look for NUMBERS in over fields 
   - **VALUES**: <integer> | "empty" | "null" 
 
9. **refused**: 
   - **KEYWORD TARGETS**: "Total Cases Rejected", "REFUSED", "ROC Damage", "ROC Damage", "R", "ROC Damage Return On Carrier" 
   - **EXTRACTION RULE**: Look for NUMBERS in refused/ROC fields 
   - **VALUES**: <integer> | "empty" | "null"

### CRITICAL VALIDATION RULES: 
- **Single Letter Field Detection**: 
  - "D" field = damage quantity (look for number next to/below "D") 
  - "S" field = short quantity (look for number next to/below "S") 
  - "O" field = over quantity (look for number next to/below "O") 
  - "R" field = refused quantity (look for number next to/below "R") 
- **Number Extraction from Single Letter Fields**: Extract the HANDWRITTEN NUMBER associated with each letter field 
- **Field Presence Logic**: 
  - null = field doesn't exist in stamp area 
  - empty = field exists but no value written/blank 
  - <integer> = field has numeric value written 
- **Stamp Area Focus**: Only extract from receiving stamp area or "Notation Area (Customer Use ONLY)" ignore other document sections 
- **Avoid Confusion Sources**: 
  - **For total_received**: Ignore appointment numbers, pallet counts, reference numbers, trailer, seal and other numbers only consider these labels("CARTONS", "TOT CS REC", "TOTAL CARTONS", "Ctns Received", "Ctns. Received")
  - **For all fields**: Focus on stamp area only, not document body
- **SEAL INTACT STRICT RULE**: 
  - ONLY return "yes"/"no" if you find EXACTLY: "Seal(s) Intact Y/N", "SEAL INTACT Y N", or "Seal Intact? Y N"
  - If you see ANY other seal-related text (like "Seal Intact", "Seal Status", etc.), return "null"
  - This prevents false positives from similar but different fields

### Output Format:
```json
{
  "pod_date": "<string>" | "empty" | "null",
  "pod_sign": "yes" | "no" | "empty" | "null", 
  "stamp_exist": "yes" | "no",
  "seal_intact": "yes" | "no" | "empty" | "null",
  "damage": <integer> | "empty" | "null",
  "short": <integer> | "empty" | "null", 
  "over": <integer> | "empty" | "null",
  "refused": <integer> | "empty" | "null",
  "total_received": <integer> | "empty" | "null",
  "roc_damaged": <integer> | "empty" | "null",
  "damaged_kept": <integer> | "empty" | "null"
}
```

**Analysis Process:**
1. First, identify if the receiving stamp area or "Notation Area (Customer Use ONLY)" exists
2. For each field, determine: Is field present? What value is written?
3. Distinguish between handwritten numbers and letter labels (S, D, O, R)
4. Provide detailed reasoning for each extraction
5. Generate final JSON
"""

DELIVERY_RECEIPT_PROMPT = """
**Delivery Receipt Analysis - Enhanced Accuracy**

Extract information from delivery receipt documents with precise field identification.

### Key Extraction Rules:

1. **`pod_date`**:
   - **Target locations**: Date fields inside "RECEIVING STAMP AREA" or "Notation Area (Customer Use ONLY)" area
   - **Include**: Receipt dates, delivery dates, confirmation dates
   - **Exclude**: "Avail Date", "Sched Date", "Date Unloaded" 
   - **Format**: Preserve original format exactly
   - **Values**: "<date_string>" | "empty" | "null"

2. **`pod_sign`**:
   - **Target fields**: "Target Signature", "RECVR", receiver signature areas
   - **Signature detection**: Any handwritten marks, scribbles, or initials
   - **Exclude**: "Driver Signature" fields
   - **Values**: "yes" (signed) | "no" (unsigned) | "empty" (blank field) | "null" (no field)

3. **`total_received`**:
   - **Target fields**: "CARTONS", "TOT CS REC", "TOTAL CARTONS", "Ctns Received", "Ctns. Received"
   - **Extraction**: Handwritten numbers in received quantity fields
   - **Exclude**: "SEAL #'S", "APPOINTMENT #", "TRAILER #'S", "Pallets", "PALLETS", "PLTS", "TOT CS PLT", "TOT CS PLTS"
   - **Multiple values**: If multiple line items, extract individual numbers
   - **Values**: <integer> | [<integer>, <integer>] | "empty" | "null"

4. **`damage`**:
   - **Target fields**: "External Dam'g Recv'd", "Damaged Received"
   - **Extraction**: Handwritten damage quantities
   - **Values**: <integer> | [<integer>, <integer>] | "empty" | "null"

5. **`refused`**:
   - **Target fields**: "Returned to Carrier", "Refused"
   - **Extraction**: Quantities returned/refused
   - **Values**: <integer> | [<integer>, <integer>] | "empty" | "null"

6. **`customer_number`**:
   - **Target locations**: "Src / PO" column, "IN TIME"/"OUT TIME" sections
   - **Criteria**: Numbers 10-15 digits long, may contain "/"
   - **Multiple values**: Extract all, separate with commas
   - **Values**: "<string>" | "<string1>,<string2>" | "null"

### Output Format:
```json
{
  "pod_date": "<string>" | "empty" | "null",
  "pod_sign": "yes" | "no" | "empty" | "null", 
  "total_received": <integer> | [<integer>] | "empty" | "null",
  "damage": <integer> | [<integer>] | "empty" | "null",
  "refused": <integer> | [<integer>] | "empty" | "null",
  "customer_number": "<string>" | "null"
}
```

First analyze each target field location and content, then generate JSON output.
"""

RECEIPT_DATE_PROMPT = """
**Receipt Date Extraction - High Precision**

Extract the primary date from the receipt document.

**Key: `pod_date`**:
- **Location**: inside the "RECEIVING STAMP AREA" or "Notation Area (Customer Use ONLY)" area
- **Format preservation**: Extract exactly as written (MM/DD/YYYY, MM/DD/YY, MM-DD-YYYY, etc.)
- **Exclusions**: Ignore "Avail Date", "Sched Date", "Date Unloaded", appointment dates
- **Partial dates**: If some digits unclear, extract visible portion
- **Values**: "<date_string>" | "empty" (field present but blank) | "null" (no date field)

**Output Format:**
```json
{
  "pod_date": "<string>" | "empty" | "null"
}
```

First locate and read the date field, then generate JSON output.
"""

RECEIPT_SIGNATURE_PROMPT = """
**Signature Detection - Enhanced Precision**

Detect handwritten signatures in receiver fields.

**Key: `pod_sign`**:
- **Target fields**: "Target Signature", "RECVR", receiver signature areas
- **Signature criteria**: Any handwritten marks, scribbles, initials, or cursive text
- **Include**: Even brief handwritten marks that indicate signing
- **Exclude**: "Driver Signature", printed names only
- **Field states**: 
  - Present with handwritten content: "yes"
  - Present but unsigned: "no" 
  - Present but blank: "empty"
  - Field not found: "null"

**Output Format:**
```json
{
  "pod_sign": "yes" | "no" | "empty" | "null"
}
```

First examine signature fields for handwritten content, then generate JSON output.
"""

RECEIPT_TOTAL_RECEIVED_PROMPT = """
**Total Received Quantity Extraction**

Extract received quantities from receipt tables.

**Key: `total_received`**:
- **Target column**: "CARTONS", "TOT CS REC", "TOTAL CARTONS", "Ctns Received", "Ctns. Received"
- **Extraction method**: List all integer values from the received column only consider these labels("CARTONS", "TOT CS REC", "TOTAL CARTONS", "Ctns Received", "Ctns. Received")
- **Exclude columns**: "Sched" (scheduled quantities) , "PLTS", "TOT CS PLT", "TOT CS PLTS", "Appointment #", "Trailer #", "TRAILER #'S", SEAL #'S "Trailer", "zip code" 
- **Multiple rows**: Extract each row's received quantity separately
- **Values**: [<int1>, <int2>, ...] | "empty" (column present, no values) | "null" (no column)

**Output Format:**
```json
{
  "total_received": [<integer>] | "empty" | "null"
}
```

First locate received quantity column and extract all values, then generate JSON output.
"""

RECEIPT_DAMAGE_PROMPT = """
**Damage Quantity Extraction**

Extract damage quantities from receipt tables.

**Key: `damage`**:
- **Target column**: "External Dam'g Rcv'd", damage-related columns
- **Extraction method**: List all integer values from damage columns
- **Exclude**: "Returned to Carrier" columns
- **Multiple rows**: Extract each damage quantity separately
- **Values**: [<int1>, <int2>, ...] | "empty" (column present, no values)

**Output Format:**
```json
{
  "damage": [<integer>] | "empty"
}
```

First locate damage columns and extract all integer values, then generate JSON output.
"""

RECEIPT_REFUSED_PROMPT = """
**Refused/Returned Items Extraction**

Extract quantities returned to carrier from table.

**Task**: Analyze table rows for "Returned to Carrier" information.

**Key: `refused`**:
- **Target column**: "Returned to Carrier" or similar
- **Extraction**: All integers written in the refused/returned column
- **Row analysis**: Each number represents items returned for that row
- **Handwritten/printed**: Accept both handwritten and printed numbers
- **Values**: [<int1>, <int2>, ...] | "empty" (column present, no values)

**Output Format:**
```json
{
  "text": "<all_visible_text_in_image>",
  "refused": [<integer>] | "empty"
}
```

First identify the returned to carrier column, extract all integers, then generate JSON with reasoning.
"""

RECEIPT_CUSTOMER_ORDER_NUMBER_PROMPT = """
**Customer Order Number Extraction - Enhanced Detection**

Extract customer order numbers based on receipt format type.

**Key: `customer_order_num`**:

### Short Receipts:
- **Target fields**: Numbers under/near "IN TIME", "OUT TIME" labels
- **Characteristics**: ~12 digit numbers (minimum 10 digits)
- **Layout**: May be vertical or horizontal arrangement
- **Multiple values**: Extract all customer numbers found

### Long Receipts:  
- **Target column**: "SRC / PO" column
- **Format**: Numbers with "/" separator (extract part after "/")
- **Multiple rows**: Extract from each row with customer data

### General Rules:
- **Length filter**: Only numbers ≥10 digits (customer order numbers are long)
- **Exclusions**: Skip numbers with "-" (hyphen) like "123415-12"
- **Format**: Clean numbers, preserve "/" if part of customer format
- **Multiple values**: Return as array if multiple numbers found

**Output Format:**
```json
{
  "customer_order_num": ["<string1>", "<string2>"] | "<string>" | "null"
}
```

First identify receipt type and locate customer number fields, then generate JSON output.
"""