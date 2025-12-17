"""
prompts.py
"""

SIGNATURE_PROMPT = """
Analyze the image to identify handwritten signatures.

**Key: `receiver_signature`**
- Look in: "RECEIVING STAMP AREA" or "Notation Area (Customer Use ONLY)"
- Check fields: "SIGNATURE", "REC'D BY", "RECEIVED BY", "PRINT NAME"
- **Important**: Only handwritten marks/scribbles count as "yes"
- Printed names/stamps without handwriting = "no"

**Values:**
- `"yes"`: Handwritten signature clearly present
- `"no"`: No handwritten signature

**CRITICAL OUTPUT INSTRUCTIONS:**
Return ONLY valid JSON without any markdown formatting, code blocks, or additional text.
Do NOT wrap the response in ```json or any other markers.

**Output Format:**
{
  "receiver_signature": "yes"
}

OR

{
  "receiver_signature": "no"
}

Analyze signature fields, then output raw JSON only - no markdown, no code blocks, no explanations.
"""

BILL_OF_LADING_PROMPT = """
Extract bill number and date from Bill of Lading document.

**Extraction Rules:**

1. **`bill_no`**: 
   - Find: "Bill of Lading:" label + number (8-9 digits)
   - Extract ONLY the numeric value
   - Example: "Bill of Lading: 17540697" → "17540697"

2. **`date`**:
   - ONLY from: "RECEIVING STAMP AREA" or "Notation Area (Customer Use ONLY)"
   - Extract exactly as written: "10/23/24", "10-23-2024", etc.
   - Ignore: header dates, appointment dates, delivery dates
   - Partial visibility: extract visible portion
   - Completely unreadable: "null"

**Output Format:**
```json
{
  "bill_no": "<string>",
  "date": "<string>" | "null"
}
```

Locate bill number and date, then output JSON only.
"""

CUSTOMER_ORDER_INFO_PROMPT = """
Extract total order quantity - NO CALCULATIONS ALLOWED.

**Key: `total_order_quantity`**

**CRITICAL: DO NOT CALCULATE**
- Find pre-calculated total ONLY
- Do NOT sum line items
- Do NOT multiply values
- Do NOT add columns

**Priority:**
1. "TOTAL QTY", "TOTAL QUANTITY" fields
2. Single "QTY" value (not multiple rows)
3. FIRST quantity value if multiple exist

**Locations:**
- "CUSTOMER ORDER INFORMATION" section totals
- Pre-calculated summary fields

**Avoid:**
- Summing rows: QTY: 34, QTY: 35 → use 34 only
- Multiplying: 2 x 34 → use 34 only
- Adding columns: QTY + HANDLING → use QTY only

**Output Format:**
```json
{
  "total_order_quantity": <integer>
}
```

Find total quantity value, then output JSON only.
"""

STAMP_LATEST_PROMPT = """
STAMP EXTRACTION - PRECISION MODE

Extract data from RECEIVING STAMP or NOTATION AREA.

SEARCH ZONES (check BOTH):
1. "RECEIVING STAMP AREA"
2. "Notation Area (Customer Use ONLY)"

═══════════════════════════════════════════════════════════════

EXTRACTION RULES:

1. **stamp_exist**:
   • Look for: stamp impressions, ink marks, rectangular/circular stamps
   • "yes" = any stamp visible
   • "no" = no stamp

2. **seal_intact**:
   • ONLY if EXACT text: "Seal(s) Intact Y/N" or "SEAL INTACT Y N"
   • Check marked letter: Y or N
   • "yes" = Y marked/circled
   • "no" = N marked/circled
   • "empty" = field exists, nothing marked
   • "null" = field does NOT exist

3. **pod_date**:
   • Find: "DATE", "Signed Date", Pick Up Appointment Date & Time, date next to signature
   • Must be in stamp/notation area (not header)
   • "empty" = field exists but blank
   • "null" = field does NOT exist

4. **pod_sign**:
   • Find: "SIGNATURE", "RECEIVED BY", "REC'D BY", "PRINT NAME"
   • "yes" = any handwritten marks/scribbles
   • "no" = only printed text/stamps
   • "empty" = field exists but blank
   • "null" = field does NOT exist

5. **total_received** (CARTONS ONLY):

   CRITICAL: DISTINGUISH BETWEEN PALLETS AND CARTONS
   
   STEP 1 - Identify what type of label you see:
   
   A) CARTON LABELS (what we WANT):
      • "TOT CS REC" or "TOTAL CS REC"
      • "CARTONS" or "TOTAL CARTONS"
      • "Ctns Received" or "Ctns. Received"
      • "Ctns Delivered" or "# Ctns Delivered"
      • "CASES RECEIVED"
   
   B) PALLET LABELS (what to IGNORE):
      • "TOT PLTS" or "TOTAL PLTS"
      • "PALLETS" or "TOT PALLETS"
      • "PLTS" or "PLT"
      • Any label containing "PALLET" or "PLT"
   
   STEP 2 - VALIDATION CHECK (Use number value to verify):
   • If unclear, look at the actual text more carefully
   
   STEP 3 - When you see "TOT CS REC" or similar:
   • First verify the number next to it
   • If number is small (1-20), re-examine if this might actually be "TOT PLTS"
   • Double-check the letters: Does it really say "CS"
   
   STEP 4 - Extract ONLY if you confirmed it's CARTONS:
   • Take the number next to the CARTON label
   • ALWAYS take the FIRST number (leftmost) before any text/symbols
   
   STEP 5 - ALWAYS IGNORE:
   • Any number next to "TOT PLTS", "PALLETS", "PLT"
   • "APPT", "APPOINTMENT #"
   • "TRAILER #", "UNIT #"
   • "SEAL #"
   • 4-digit numbers (times: 1504, 1430)
   • DC numbers (DC1234)
   
   OUTPUT:
   • <integer> = carton number found AND VERIFIED
   • "empty" = carton label exists but no number written
   • "null" = carton label field does NOT exist

6. **damage**:
   • Find: "DAMAGED", "Damage Kept", "Damage", "Cartons Damaged", "# Ctns Damaged", "D"
   • Extract: integer only
   • "empty" = field exists but blank
   • "null" = field does NOT exist

7. **short**:
   • Find: "SHORT", "Cartons Short", "# Ctns Short", "S"
   • Extract: integer only
   • "empty" = field exists but blank
   • "null" = field does NOT exist

8. **over**:
   • Find: "OVER", "# Ctns Over", "O"
   • Extract: integer only
   • "empty" = field exists but blank
   • "null" = field does NOT exist

9. **refused**:
   • Find: "REFUSED", "ROC Damage", "Total Cases Rejected", "R"
   • Extract: integer only
   • "empty" = field exists but blank
   • "null" = field does NOT exist

10. **roc_damaged**:
    • Find: "ROC Damage", "ROC DAMAGED"
    • Extract: integer only
    • "empty" = field exists but blank
    • "null" = field does NOT exist

11. **damaged_kept**:
    • Find: "Damage Kept", "DAMAGED KEPT"
    • Extract: integer only
    • "empty" = field exists but blank
    • "null" = field does NOT exist

12. **notation_exist**:
   CRITICAL: Look for the EXACT text "Notation Area (Customer Use ONLY)" or "Notation Area (Customer Use Only)"
   
   This is a specific header text, NOT a general section.
   
   MUST SEE:
   • The exact words: "Notation Area"
   • Followed by: "(Customer Use ONLY)" or "(Customer Use Only)"
   • This appears as a header/title in the document
   
   DO NOT confuse with:
   • Regular receiving stamp areas
   • Customer information sections
   • Any section that just has customer fields
   • Stamp boxes without "Notation Area" text
   
   DECISION RULE:
   • Can you see the text "Notation Area" anywhere? 
     - YES → return "yes"
     - NO → return "no"
   
   OUTPUT:
   • "yes" = Text "Notation Area (Customer Use ONLY)" is clearly visible
   • "no" = This specific text is NOT present (default to "no" if unsure)
   
   IMPORTANT: Default to "no" unless you are 100% certain you see "Notation Area" text

═══════════════════════════════════════════════════════════════

CRITICAL RULES:
• Focus ONLY on stamp/notation areas
• Extract numbers, NOT formulas (34, not "2x34")
• "null" = field does NOT exist
• "empty" = field exists but has no value/blank
• <value> = actual data found

**CRITICAL OUTPUT INSTRUCTIONS:**
Return ONLY valid JSON without any markdown formatting, code blocks, or additional text.
Do NOT wrap the response in ```json, ```java, ```javascript or any other markers.
Do NOT include explanations, descriptions, or any text before or after the JSON.

OUTPUT FORMAT (raw JSON only):
{
  "stamp_exist": "yes",
  "seal_intact": "yes",
  "pod_date": "<string>" | "empty" | "null",
  "pod_sign": "yes",
  "total_received": <integer> | "empty" | "null",
  "damage": 0,
  "short": 0,
  "over": 0,
  "refused": 0,
  "roc_damaged": 0,
  "damaged_kept": 0,
  "notation_exist": "no"
}

Think step-by-step, then output raw JSON only - no markdown, no code blocks, no explanations.
"""

DELIVERY_RECEIPT_PROMPT = """
Extract information from delivery receipt document.

EXTRACTION RULES:

1. **pod_date**:
   • Target: date fields in "RECEIVING STAMP AREA" or "Notation Area (Customer Use ONLY)"
   • Include: receipt dates, delivery dates, confirmation dates
   • Exclude: "Avail Date", "Sched Date", "Date Unloaded"
   • Format: preserve exactly as written
   • "<date_string>" | "empty" | "null"

2. **pod_sign**:
   • Target: "Target Signature", "RECVR", receiver signature areas
   • Detect: handwritten marks, scribbles, initials
   • Exclude: "Driver Signature"
   • "yes" | "no" | "empty" | "null"

3. **total_received**:
   • Target: "CARTONS", "TOT CS REC", "TOTAL CARTONS", "Ctns. Received", "Ctns Received", "Ctns Delivered"
   • Extract: handwritten numbers only
   • Exclude: "SEAL #'S", "APPOINTMENT #", "TRAILER #'S", "PALLETS", "PLTS"
   • Multiple values: extract individually
   • <integer> | [<integer>] | "empty" | "null"

4. **damage**:
   • Target: "External Dam'g Recv'd", "Damaged Received"
   • Extract: damage quantities
   • <integer> | [<integer>] | "empty" | "null"

5. **refused**:
   • Target: "Returned to Carrier", "Refused"
   • Extract: returned/refused quantities
   • <integer> | [<integer>] | "empty" | "null"

6. **customer_order_num**:
   • Target: "Src / PO" column, "IN TIME"/"OUT TIME" sections
   • Criteria: 10-15 digit numbers, may contain "/"
   • Multiple: extract all
   • "<string>" | ["<string>"] | "null"

CRITICAL RULES:
• "null" = field does NOT exist
• "empty" = field exists but has no value/blank
• <value> = actual data found

**CRITICAL OUTPUT INSTRUCTIONS:**
Return ONLY valid JSON without any markdown formatting, code blocks, or additional text.
Do NOT wrap the response in ```json, ```java, ```javascript or any other markers.

OUTPUT FORMAT (raw JSON only):
{
  "pod_date": "<string>" | "empty" | "null",
  "pod_sign": "yes",
  "total_received":  <integer> | [<integer>] | "empty" | "null",
  "damage": 0,
  "refused": 0,
  "customer_order_num": "<string>" | ["<string>"] | "null"
}

Analyze fields, then output raw JSON only - no markdown, no code blocks, no explanations.
"""

RECEIPT_DATE_PROMPT = """
Extract date from receipt document.

**pod_date**:
• Location: "RECEIVING STAMP AREA" or "Notation Area (Customer Use ONLY)"
• Format: extract exactly as written (MM/DD/YYYY, MM/DD/YY, MM-DD-YYYY)
• Exclude: "Avail Date", "Sched Date", "Date Unloaded", appointment dates
• Partial dates: extract visible portion
• "<date_string>" | "empty" | "null"

OUTPUT FORMAT:
```json
{
  "pod_date": "<string>" | "empty" | "null"
}
```

Locate date field, then output JSON only.
"""

RECEIPT_SIGNATURE_PROMPT = """
Detect handwritten signature in receiver fields.

**pod_sign**:
• Target: "Target Signature", "RECVR", receiver signature areas
• Criteria: handwritten marks, scribbles, initials, cursive
• Include: even brief handwritten marks
• Exclude: "Driver Signature", printed names only
• "yes" = present with handwriting
• "no" = present but unsigned
• "empty" = present but blank
• "null" = field not found

OUTPUT FORMAT:
```json
{
  "pod_sign": "yes" | "no" | "empty" | "null"
}
```

Examine signature fields, then output JSON only.
"""

RECEIPT_TOTAL_RECEIVED_PROMPT = """
Extract received quantities from receipt tables.

**total_received**:
• Target: "CARTONS", "TOT CS REC", "TOTAL CARTONS", "Ctns. Received", "Ctns Received", "Ctns Delivered"
• Method: list all integers from received column
• Exclude: "Sched", "PLTS", "TOT CS PLT", "Appointment #", "Trailer #", "SEAL #'S"
• Multiple rows: extract each separately
• [<integer>] | "empty" | "null"

OUTPUT FORMAT:
```json
{
  "total_received": [<integer>] | "empty" | "null"
}
```

Locate received column, then output JSON only.
"""

RECEIPT_DAMAGE_PROMPT = """
Extract damage quantities from receipt tables.

**damage**:
• Target: "External Dam'g Rcv'd", damage-related columns
• Method: list all integers from damage columns
• Exclude: "Returned to Carrier"
• Multiple rows: extract each separately
• [<integer>] | "empty"

OUTPUT FORMAT:
```json
{
  "damage": [<integer>] | "empty"
}
```

Locate damage columns, then output JSON only.
"""

RECEIPT_REFUSED_PROMPT = """
Extract returned/refused quantities from table.

**refused**:
• Target: "Returned to Carrier" or similar columns
• Extract: all integers in refused/returned column
• Each number = items returned for that row
• Accept: handwritten and printed numbers
• [<integer>] | "empty"

OUTPUT FORMAT:
```json
{
  "refused": [<integer>] | "empty"
}
```

Identify returned column, then output JSON only.
"""

RECEIPT_CUSTOMER_ORDER_NUMBER_PROMPT = """
Extract customer order numbers from receipt.

**customer_order_num**:

SHORT RECEIPTS:
• Target: numbers near "IN TIME", "OUT TIME" labels
• Characteristics: ~12 digits (minimum 10)
• Multiple: extract all found

LONG RECEIPTS:
• Target: "SRC / PO" column
• Format: numbers with "/" (extract part after "/")
• Multiple rows: extract from each

RULES:
• Length: only numbers ≥10 digits
• Exclude: numbers with "-" hyphen (123415-12)
• Format: preserve "/" if present
• Multiple: return as array

OUTPUT FORMAT:
```json
{
  "customer_order_num": ["<string>"] | "<string>" | "null"
}
```

Identify receipt type and locate customer numbers, then output JSON only.
"""