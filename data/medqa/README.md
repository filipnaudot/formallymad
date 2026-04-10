# MedQA

> Source: [jind11/MedQA](https://github.com/jind11/MedQA?tab=readme-ov-file)

MedQA is a multiple-choice question answering dataset based on the United States Medical Licensing Examination (USMLE).
Questions are sourced from Steps 1 and 2&3 of the exam and cover basic medical science and clinical reasoning respectively.

## Schema

Each record is a JSON object with the following fields:

| Field | Type | Description |
|---|---|---|
| `question` | `str` | The clinical vignette or question stem |
| `options` | `dict[str, str]` | Four answer choices keyed A–D |
| `answer` | `str` | The correct answer text |
| `answer_idx` | `str` | The correct answer key (A, B, C, or D) |
| `meta_info` | `str` | USMLE step: `"step1"` or `"step2&3"` |
| `metamap_phrases` | `list[str]` | Medical concepts extracted via MetaMap |

## Files

| File | Questions | Description |
|---|---|---|
| `phrases_no_exclude_dev.jsonl` | 1,272 | Full development split (701 Step 1, 571 Step 2&3) |
| `medqa_30.jsonl` | 30 | Reduced benchmark subset (see below) |

## medqa_30.jsonl

A 30-question subset created for benchmarking, drawn from the dev split:

- First 15 **Step 1** questions (basic science)
- First 15 **Step 2&3** questions (clinical reasoning)

Step 2 and Step 3 are not distinguished in the source data and appear combined under the `"step2&3"` label.
