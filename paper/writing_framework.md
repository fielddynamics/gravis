# DFP Writing Framework

**Purpose:** A unified pipeline for writing physics papers that combines pedagogical excellence with formatting compliance. Every piece of content flows through this pipeline to ensure clarity, structure, and professional output.

---

## The Core Problem This Framework Solves

| Problem | Symptom | Solution |
|---------|---------|----------|
| "Textbook mode" | Output includes "Learning Objectives" boxes, "What you will learn" | Strict internal/external distinction |
| Formatting drift | Callouts stacked, tables in boxes, em dashes | Mandatory Stage 4 compliance check |
| Context loss | Subsections don't connect | Stage 1 requires reading before/after |
| Unclear writing | How-before-why, defensive tone | Stage 3 writing principles |

---

## The Four-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         WRITING FRAMEWORK                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   STAGE 1              STAGE 2              STAGE 3              STAGE 4 │
│   ─────────            ─────────            ─────────            ─────── │
│   UNDERSTAND           PLAN                 WRITE                FORMAT  │
│   (Read context)       (Internal only)      (Apply principles)   (Comply)│
│                                                                          │
│   ↓ What exists?       ↓ What to achieve?   ↓ How to explain?    ↓ Rules │
│   ↓ Before/after?      ↓ Core concepts?     ↓ Why before how?    ↓ Check │
│   ↓ Reader state?      ↓ Exit criteria?     ↓ Teach, not defend  ↓ Fix   │
│                                                                          │
│   OUTPUT: Context      OUTPUT: Internal     OUTPUT: Draft        OUTPUT: │
│           notes        plan (never          prose/content        Final   │
│                        published)                                content │
└─────────────────────────────────────────────────────────────────────────┘
```

**Critical Rule:** Stage 2 outputs are NEVER published. They inform Stage 3 writing.

---

# STAGE 1: UNDERSTAND

**Goal:** Build complete context before writing anything.

## 1.1 Required Reading

Before writing or editing ANY section/subsection, you MUST read:

| What | Why |
|------|-----|
| **The full section** | Understand the container |
| **Previous section/subsection** | What does the reader know coming in? |
| **Next section/subsection** | What must we prepare them for? |
| **Related content** | Cross-references, callbacks |

**Hard Rule:** Never write a subsection without knowing its before and after.

## 1.2 Context Capture Template

```
SECTION: [Number and title]
SUBSECTION: [Number and title]

READER STATE ENTERING:
- Knows: [What the reader understands at this point]
- Expects: [What question they're ready to ask]

READER STATE EXITING:
- Knows: [What they will understand after]
- Ready for: [What question they can now tackle]

CONNECTS TO:
- Previous: [Explicit connection to prior content]
- Next: [What this prepares them for]
```

## 1.3 Stage 1 Checklist

- [ ] Read the full section containing this content
- [ ] Read what comes before (previous subsection or section)
- [ ] Read what comes after (next subsection or section)
- [ ] Documented reader entry state
- [ ] Documented reader exit state
- [ ] Identified explicit connections

**Do NOT proceed to Stage 2 until Stage 1 is complete.**

---

# STAGE 2: PLAN (Internal Only)

**Goal:** Define what the content must achieve. This stage is for YOUR planning only.

## ⚠️ CRITICAL: Stage 2 Outputs Are Never Published

| Stage 2 Element | In Published Paper? |
|-----------------|---------------------|
| Learning Objectives | ❌ NEVER |
| Core Concepts list | ❌ NEVER |
| Exit Criteria | ❌ NEVER |
| Internal notes | ❌ NEVER |
| Reader journey map | ❌ NEVER |

These are thinking tools. They guide writing but do not appear in output.

## 2.1 Section/Subsection Plan Template

```
═══════════════════════════════════════════════════════════════
INTERNAL PLANNING (DO NOT PUBLISH)
═══════════════════════════════════════════════════════════════

PURPOSE: [One sentence - what this section accomplishes]

LEARNING OBJECTIVES (internal use only):
1. Reader will understand [concept 1]
2. Reader will understand [concept 2]
3. Reader will be able to [skill/application]

CORE CONCEPTS:
1. [First essential idea]
2. [Second essential idea]
3. [Third essential idea]

EXIT CRITERIA:
- Reader understands: [concrete outcome]
- Reader is ready for: [next topic]

VISUAL ELEMENT NEEDED: [Analogy, diagram, example, or table]

TENSION ARC: [What question drives this section?]

═══════════════════════════════════════════════════════════════
END INTERNAL PLANNING
═══════════════════════════════════════════════════════════════
```

## 2.2 How to Transform Internal Plan to Published Content

**Learning Objectives → Section Opening Paragraph**

❌ NEVER publish:
```html
<div class="highlight-box info">
    <h4>Learning Objectives</h4>
    <p>By the end of this section, you will:</p>
    <ul>
        <li>Derive r_e from electromagnetic stability</li>
        <li>Understand why convergence is significant</li>
    </ul>
</div>
```

✅ Transform to:
```html
<p>
    This section derives the characteristic tile size r_e = 2.82 fm through 
    electromagnetic stability, then explains why the convergence of two 
    independent routes to this scale is physically significant.
</p>
```

**Core Concepts → Structure the subsections around them**

**Exit Criteria → Verify the content delivers them**

## 2.3 Stage 2 Checklist

- [ ] Defined PURPOSE (one sentence)
- [ ] Listed internal Learning Objectives (will NOT publish)
- [ ] Identified 3 Core Concepts
- [ ] Set Exit Criteria
- [ ] Planned visual element
- [ ] Identified tension arc / driving question

**Do NOT proceed to Stage 3 until Stage 2 is complete.**

---

# STAGE 3: WRITE

**Goal:** Produce clear, pedagogically excellent prose that teaches.

## 3.1 The Wesley Crusher Standard

The goal: exposition that belongs in a 24th-century engineering textbook because it's simply the clearest explanation ever written.

| Quality | What It Means |
|---------|---------------|
| **Easy to read** | No unnecessary friction. Every sentence earns its place. |
| **Logically coherent** | Each idea follows from the previous. No jumps. |
| **Elegant simplicity** | Complex ideas made clear, not simple ideas made complex. |
| **Everything makes sense** | No "just accept this" moments. Every claim is grounded. |
| **Concepts build** | Earlier sections prepare you for later ones. |
| **Clean** | No clutter. No redundancy. No tangents. |

**The Test:** After reading, a smart reader should feel: "I understand this. I could explain it to someone else."

## 3.2 Writing Principles

### Principle 1: Why Before How

**Research shows:** Students who learn *why* before *how* demonstrate better conceptual understanding.

❌ Before:
```
The wave equation is:
∂²h/∂t² = c²∇²h
This describes gravitational wave propagation.
```

✅ After:
```
How do disturbances propagate through a field? The answer lies in the 
relationship between how the field changes in time and how it varies in space:

∂²h/∂t² = c²∇²h

The constant c sets the propagation speed—for gravity, the speed of light.
```

### Principle 2: Teach, Don't Defend

❌ Defensive framing:
- "This cannot be disputed"
- "A referee who accepts X must accept Y"
- "There is no hypothesis to reject"

✅ Teaching framing:
- "Having established X, we now ask Y"
- "This follows from the definition of..."
- "We can trace the logical chain..."

### Principle 3: Confident, Not Defensive

| Defensive | Confident |
|-----------|-----------|
| "One cannot deny that..." | "This follows from..." |
| "Critics might object, but..." | (Don't mention critics) |
| "This is undeniable" | "This completes the argument" |

### Principle 4: The Equation Sandwich

Every equation needs:
```
[MOTIVATION - why we need this]
     ↓
[EQUATION]
     ↓
[INTERPRETATION - what it tells us]
```

### Principle 5: Visualization

Every abstract concept benefits from a concrete image:
- **Abstract:** Gravity is a distributed field
- **Concrete:** Ocean waves propagate because molecules push neighbors

### Principle 6: Academic Register (Referee-Safe Language)

**RULE:** All language must be referee-safe. Informal phrases—even when accurate—undermine credibility and trigger rejection.

#### Informal → Academic Conversion Table

| NOT Referee-Safe | Referee-Safe Equivalent | Category |
|------------------|-------------------------|----------|
| unassailable | well-established in the literature | Defensive |
| fortress | established foundation | Metaphor |
| breakthrough | resolution, key development | Press-release |
| dramatic | notable, significant | Emotional |
| fatal disease | fundamental inconsistency | Metaphor |
| sick (theory is sick) | mathematically inconsistent | Colloquial |
| seemed impossible | appeared intractable | Imprecise |
| extraordinary | significant | Hyperbolic |
| slapped onto | assigned to | Colloquial |
| tried | attempted | Informal |
| fix / fixed | resolve / resolved | Informal |
| found / discovered | demonstrated, established | Imprecise |
| game-changer | significant advance | Journalistic |
| bulletproof / ironclad | rigorously established | Metaphor |
| amazing / incredible | noteworthy | Hyperbolic |
| obvious / clearly | follows directly, is evident | Dismissive |

#### Actual Corrections From Paper VI

These are real before/after fixes:

| Section | NOT Referee-Safe (Original) | Referee-Safe (Corrected) |
|---------|----------------------------|--------------------------|
| §1.3 | "Four roads to two tetrahedra (the fortress)" | "Four roads to two tetrahedra (the established foundation)" |
| §2 box | "This foundation is unassailable" | "This foundation is well-established in the literature" |
| §2.1 | "making the whole theory sick" | "rendering the theory mathematically inconsistent" |
| §2.4 | "This road has a dramatic history" | "This road has a notable history" |
| §2.4 | "developed a fatal disease" | "developed a fundamental inconsistency" |
| §2.4 | "seemed impossible" | "appeared intractable" |
| §2.4 | "The breakthrough" | "The resolution" |
| §2.4 | "discovered that promoting..." | "demonstrated that promoting..." |
| §6.1 | "that is extraordinary" | "that is significant" |
| §6.1 | "slapped onto particles" | "assigned to particles" |

#### Self-Check Before Submission

Scan every section for:

1. **Metaphors** — fortress, battlefield, disease, ammunition, weapon
2. **Superlatives** — amazing, incredible, extraordinary, stunning, powerful
3. **Colloquialisms** — slam dunk, no-brainer, game-changer, sick
4. **Emotional language** — dramatic, exciting, thrilling, remarkable
5. **Defensive language** — unassailable, bulletproof, ironclad, airtight
6. **Press-release language** — breakthrough, revolutionary, paradigm-shifting

**The Standard:** If you would not see it in *Physical Review Letters*, do not use it.

## 3.3 Subsection Structure

Every subsection MUST have:

1. **Opening Paragraph**
   - Never start with an equation
   - State what this subsection addresses
   - Connect to what came before
   - (This is where Learning Objectives become implicit)

2. **The WHY Before the HOW**
   - Why are we doing this?
   - What question are we answering?
   - THEN: the method/equation/derivation

3. **Clear Progression**
   - Each paragraph builds on the previous
   - Signpost transitions: "Having established X, we now..."

4. **Closing That Connects Forward**
   - What did we establish?
   - What does this prepare us for?

## 3.4 Stage 3 Checklist

- [ ] Opens with context paragraph (not equation)
- [ ] WHY is explained before HOW
- [ ] Every equation has motivation → equation → interpretation
- [ ] No defensive language
- [ ] No informal language (see Principle 6: Academic Register)
- [ ] Connects to previous content
- [ ] Prepares for next content
- [ ] Has visual element or concrete example
- [ ] Reader could explain the content afterward

**Do NOT proceed to Stage 4 until Stage 3 is complete.**

---

# STAGE 4: FORMAT (Compliance Check)

**Goal:** Ensure all content complies with formatting guidelines.

## 4.1 Hard Rules (Must Pass)

### Rule 1: No Em Dashes in Prose
| ❌ Never | ✅ Use Instead |
|----------|----------------|
| "gravity — a field" | "gravity, a field" |
| "The answer — surprisingly — is four" | "The answer, surprisingly, is four" |

### Rule 2: No Stacked Callouts
**Max 2 callouts in sequence.** If you have 3+, convert one to prose.

### Rule 3: Section Opening Structure
```html
<h2>X. Section Title</h2>
<p>Brief intro paragraph. (1-2 paragraphs max, NO callouts here)</p>
<h3>X.1 First Subsection</h3>
```

**FORBIDDEN between h2 and h3:**
- ❌ Callouts
- ❌ Lists or bullet points
- ❌ Equations
- ❌ More than 2 paragraphs

### Rule 4: Tables Never Inside Callouts
```html
<!-- ❌ WRONG -->
<div class="highlight-box">
    <table>...</table>
</div>

<!-- ✅ CORRECT -->
<div class="table-box">
    <table>...</table>
    <caption>...</caption>
</div>
```

### Rule 5: Callout Content Limits

| Type | Max Words | Max Paragraphs |
|------|-----------|----------------|
| Definition | 100 | 1 |
| Running Example | 150 | 2 |
| Warning/Scope | 75 | 1 |
| Summary | N/A | 1 (with bullets) |

## 4.2 Callout Tiers

**MUST be a callout (Tier A):**
- Formal statements (Postulates, Theorems)
- Canonical definitions
- Paper scope declarations
- Section summaries
- Hero result

**MAY be a callout (Tier B):**
- Running Examples (max 1-2 per subsection)
- Reference tables
- Critical warnings

**MUST NOT be a callout (Tier C):**
- ❌ Learning Objectives (delete entirely)
- ❌ "Why X?" explanations → bold inline header
- ❌ Physical interpretations → inline prose
- ❌ Methodology notes → inline prose

## 4.3 CSS Class Reference

| Content Type | CSS Class | Notes |
|--------------|-----------|-------|
| Hero result (μ=x/(1+x)) | `.result` | ONE in entire paper |
| Key equations | `.equation-box` | No `\boxed{}` inside |
| Regular equations | `.equation` | Centered, no decoration |
| Callouts | `.highlight-box` | + variant: `.primary`, `.warning`, `.info` |
| Tables | `.table-box` | With caption |
| Proofs | `.proof` | + variant: `.lemma`, `.theorem` |

## 4.4 Stage 4 Compliance Checklist

Run this checklist on EVERY piece of content before finalizing:

### Structure
- [ ] Section has intro paragraph between h2 and first h3
- [ ] No callouts between h2 and h3
- [ ] No more than 2 callouts in sequence

### Callouts
- [ ] No Learning Objectives boxes
- [ ] No tables inside callouts
- [ ] Each callout ≤3 paragraphs
- [ ] Each callout contains ONE main idea
- [ ] Callout titles ≤6 words, declarative (not questions)

### Style
- [ ] No em dashes (—) in prose
- [ ] No defensive language
- [ ] No informal language (Principle 6: Academic Register)
- [ ] "We" voice used consistently
- [ ] No "it can be shown that..."

### Content
- [ ] No duplication between callout and surrounding prose
- [ ] Every equation has motivation and interpretation
- [ ] Cross-references are explicit ("§X" not "as mentioned above")

---

# Quick Reference: The Complete Pipeline

## For a New Subsection

```
1. STAGE 1: UNDERSTAND
   - Read section context
   - Read previous subsection
   - Read next subsection
   - Document reader entry/exit states

2. STAGE 2: PLAN (internal)
   - Define PURPOSE
   - List Learning Objectives (internal only!)
   - Identify Core Concepts
   - Set Exit Criteria
   - Plan visual element

3. STAGE 3: WRITE
   - Opening paragraph (implicit Learning Objectives)
   - Why before How
   - Equation sandwiches
   - Confident, teaching tone
   - Connect forward

4. STAGE 4: FORMAT
   - Run compliance checklist
   - Fix any violations
   - Verify no internal planning leaked into output
```

## For Editing Existing Content

```
1. STAGE 1: UNDERSTAND
   - Read full section
   - Read before/after
   - Understand what reader knows at each point

2. STAGE 2: PLAN (internal)
   - What should this content achieve?
   - What's missing or unclear?
   - Plan improvements

3. STAGE 3: WRITE
   - Apply writing principles
   - Maintain voice consistency
   - Preserve existing good content

4. STAGE 4: FORMAT
   - Run compliance checklist
   - Ensure no new violations introduced
```

---

# Anti-Patterns to Avoid

| Anti-Pattern | Symptom | Fix |
|--------------|---------|-----|
| **Textbook Leak** | "Learning Objectives" box in output | Delete; use opening paragraph |
| **Wall of Boxes** | 3+ callouts stacked | Convert to prose |
| **How-First** | Equation appears before motivation | Add WHY paragraph |
| **Defensive Tone** | "Critics might object..." | Remove; teach instead |
| **Informal Register** | "unassailable," "fortress," "sick," "breakthrough," "dramatic" | See Principle 6; use academic equivalents |
| **Echo Content** | Same point in callout AND prose | Delete duplicate |
| **Em Dash Habit** | "The field—which is local—" | Use commas |
| **Vague Reference** | "As mentioned above" | Use "§X.Y" |
| **Table in Box** | Table inside highlight-box | Use table-box |

---

# The Quality Standard

> **The goal is not to convince skeptics. The goal is to achieve such clarity that the explanation becomes the reference—the one future textbooks cite because no one has explained it better.**

### The Three Tests

**Griffiths Test:** Could a motivated undergraduate follow this?

**Feynman Test:** Does this explain WHY, not just WHAT?

**Landau Test:** Is every word necessary?

---

## Document Metadata

- **Created:** January 28, 2026
- **Purpose:** Unified writing pipeline combining pedagogical principles with formatting compliance
- **Replaces:** Use this instead of separate `textbook_engineering_pipeline.md` and `formatting_guidelines.md`
- **Key Innovation:** Stage 2 (internal planning) is strictly separated from published output
