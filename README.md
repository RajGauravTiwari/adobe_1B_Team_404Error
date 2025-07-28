# Truly Generic Persona-Driven Document Intelligence

## Overview

A universal Ml powered document analysis system that intelligently extracts, ranks, and refines the most relevant sections from a collection of heterogeneous PDFs, based on any specified persona and job-to-be-done. Domain-agnostic, robust, and optimized for high accuracy and adaptability (research papers, reports, guides, etc.).

---

## Directory Layout

ROOT/
│
|_app
  ├─ input/
  │ ├─ input.json # Defines documents, persona, and job/task
  │ └─ *.pdf # The PDF documents to process
  │
  ├─ output/
  │ └─ output.json # Resulting top sections and extracted content
│
├─ main.py # Contains GenericDocumentIntelligence class & entrypoint
└─ Dockerfile # Container build definition

text

---

## Installation & Usage

### 1. Prerequisites

- [Docker](https://www.docker.com/get-started/)
- (Optional) Python 3.9+, if running locally (not required with Docker)

### 2. Build the Docker Image

From your project folder run:
docker build --platform linux/amd64 -t mysolution:abc123 .

text

---

### 3. Prepare Input

- Put your PDFs and `input.json` in `input/`.
- A sample input and output has been attached.
- `input.json` example:
{
"documents": [{"filename": "doc1.pdf"}, {"filename": "doc2.pdf"}],
"persona": {"role": "HR Specialist"},
"job_to_be_done": {"task": "Create fillable onboarding forms"}
}

text


---

### 4. Run the System

docker run --rm
-v C:/DOCKER_SAMPLE/ADOBE_1B/app/input:/app/input
-v C:/DOCKER_SAMPLE/ADOBE_1B/app/output:/app/output
--network none
mysolution:abc123

text
- Adjust the `-v` mount paths for your system.

---

### 5. View Results

After completion, see `output/output.json`.  
It includes:
- Metadata (documents, persona, job)
- Top relevant sections (title, document, rank, page)
- Refined subsections with concise job-adapted content

---

## Code Structure

All core logic resides in `main.py`, encapsulated in a single class:

### `GenericDocumentIntelligence`

- **extract_hierarchical_sections**  
  Parses PDFs (PyMuPDF), dynamically detects sections using font, boldness, and pattern heuristics.

- **_enhance_title_dynamically, _enhance_sections_dynamically**  
  Cleans or generates descriptive titles for each section, fixing bad headings and deducing intent from content.

- **universal_ranking**  
  Computes a composite relevance score using:
    - Semantic similarity (`sentence-transformers`, `cosine_similarity`)
    - Job/persona keyword matches
    - Section/title/document quality
    - Adaptive weights (no domain hardcoding)

- **select_adaptive_sections**  
  Picks the top diverse sections across all documents for coverage.

- **extract_relevant_subsections**  
  Extracts the most informative blocks/subsections within top-ranked sections, tailored to the persona’s job.

- **process_documents**  
  Main pipeline:
    1. Loads config and PDFs
    2. Runs all extraction, ranking, and selection steps
    3. Saves result JSON in output directory

---

## Main Technologies

- Python (for all core logic)
- Docker (for reproducible, portable, isolated execution)
- PyMuPDF (PDF extraction)
- sentence-transformers (`all-MiniLM-L6-v2`)
- scikit-learn (cosine similarity for ranking)
- numpy, logging

---

## Security & Adaptiveness

- Strictly no outbound network (via `--network none`)
- Heuristics, scoring, and semantic search are tuned to generalize across any domain,
  persona, or job scenario—no manual rules are needed for new use-cases.

---

## Troubleshooting

- Ensure all input files are readable and mounted.
- Output directory must be writable by the container.
- If PDFs are very complex or scanned, extraction quality may vary.

---

## Citation / Credits
Raj Gaurav Tiwari - 9667782966
Abhay Kumar - 785792340
Developed for the Adobe Round 1B Challenge.

---

