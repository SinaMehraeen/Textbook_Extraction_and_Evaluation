# Textbook Chapter Extractor and Evaluator

This tool extracts chapters from textbook markdown files and evaluates them against a customizable rubric using OpenRouter's LLM API. It combines both extraction and evaluation in a single efficient process.

## Features

- **Chapter Extraction**: Automatically identifies and extracts genuine textbook chapters from markdown files
- **Rubric-based Evaluation**: Evaluates each chapter against a customizable evaluation rubric
- **Performance Optimizations**:
  - API response caching to avoid reprocessing the same books
  - Progress saving to resume interrupted runs
- **Per-Book Markdown Output**: Generates detailed markdown reports with evaluation tables for each book
- **Detailed Progress Tracking**: Real-time progress bar with status updates

## Requirements

- Python 3.7+
- OpenRouter API key

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd textbook-chapter-extractor
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Setup

1. Obtain an API key from [OpenRouter](https://openrouter.ai/)
2. Set the API key as an environment variable:
   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   ```
   Or use the `--api-key` command-line argument

## Usage

### Basic Usage

```bash
python chapter_extract_and_evaluate.py
```

### Command-Line Arguments

| Argument | Default | Description |
|---------|---------|-------------|
| `--textbook-dir` | `textbooks/` | Directory containing markdown textbooks |
| `--criteria-csv` | `Evaluation_Criteria.csv` | CSV file with evaluation criteria |
| `--output` | `artifacts/` | Output directory for extracted chapters |
| `--model` | `google/gemini-2.5-flash-lite` | OpenRouter model identifier |
| `--api-key` | None | OpenRouter API key (or set OPENROUTER_API_KEY env var) |
| `--max-book-chars` | 400000 | Max characters from each book sent to the model |
| `--max-output-tokens` | 60000 | Max tokens generated per book |
| `--temperature` | 0.0 | Sampling temperature |
| `--save-raw` | False | Save raw LLM outputs for debugging |
| `--use-cache` | False | Enable caching of API responses |
| `--resume` | False | Enable progress saving to resume interrupted runs |
| `--log-level` | INFO | Logging level |

### Performance Optimization Options

#### Caching
Enable caching to avoid reprocessing the same books:
```bash
python chapter_extract_and_evaluate.py --use-cache
```

Cache files (`.pkl`) are stored in the `artifacts/cache/` directory. These files store the raw API responses to avoid reprocessing the same books repeatedly, saving API costs and speeding up re-runs.

#### Progress Saving
Enable progress saving to resume interrupted runs:
```bash
python chapter_extract_and_evaluate.py --resume
```

Progress is saved in `artifacts/progress.pkl` and tracks which books have been successfully processed.

### Input Format

#### Textbooks
Place your textbook markdown files in the `textbooks/` directory (or specify a different directory with `--textbook-dir`).

#### Evaluation Criteria
Create a CSV file with the following columns:
- `Index`: Unique identifier for each criterion
- `Category`: Category of the evaluation criterion
- `Item`: Specific evaluation item

Example `Evaluation_Criteria.csv`:
```csv
Index,Category,Item
1,Content Quality,Provides clear learning objectives
2,Content Quality,Includes relevant examples
3,Content Quality,Uses appropriate terminology
```

### Output Format

#### Per-Book Markdown Files
Each processed book generates a markdown file with:
- YAML frontmatter with metadata
- Extracted chapters with word counts
- Evaluation tables for each chapter

The files are named `<book_id>_extracted.md` and saved in the output directory.

## How It Works

1. **Textbook Discovery**: Scans the specified directory for markdown files
2. **Chapter Extraction**: Uses an LLM to identify and extract genuine textbook chapters
3. **Evaluation**: Evaluates each extracted chapter against all rubric criteria
4. **Output Generation**: Creates detailed markdown reports with evaluation tables

## License

MIT License - see LICENSE file for details.


