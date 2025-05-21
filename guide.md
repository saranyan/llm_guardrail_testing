# LLM Guardrail Testing Framework

## Getting Started

This guide will help you quickly set up and run your first tests with the LLM Guardrail Testing Framework.

### Prerequisites

- Python 3.8 or higher
- API keys for the LLMs you want to test (Claude and/or OpenAI)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/llm-guardrail-testing.git
cd llm-guardrail-testing
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up your environment**

Create a `.env` file at the root of the project with your API keys:

```
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
```

You can also copy and modify the provided `.env.example` file.

### Running Your First Test

1. **Run a basic test using the default configuration**

```bash
python -m scripts.run_all_tests --test-suite test_suites/information_boundaries.json
```

This will execute all the tests defined in the information boundaries test suite and save the results to the `results` directory.

2. **View the results**

After the test completes, you can find:
- Raw responses in `results/raw/`
- Analyzed data in `results/processed/`
- CSV files with analytics in `results/analytics/`

3. **Generate a report**

```bash
python -m scripts.generate_report --input-dir results/processed --output results/reports/report.html
```

This will create an HTML report summarizing the test results with charts and analysis.

### Creating Your Own Test Suite

1. **Start from a template**

Copy one of the existing test suite files to create your own:

```bash
cp test_suites/information_boundaries.json test_suites/my_custom_test.json
```

2. **Edit the test suite file**

Modify the JSON file to define your own test sets, prompts, and variations. The basic structure is:

```json
{
  "name": "My Custom Test",
  "description": "Description of what this test evaluates",
  "test_sets": [
    {
      "name": "My Test Set",
      "baseline_prompt": "Your baseline prompt here",
      "variations": {
        "dimension_name": [
          {
            "description": "Variation 1",
            "prompt": "First variation of the prompt"
          },
          {
            "description": "Variation 2",
            "prompt": "Second variation of the prompt"
          }
        ]
      },
      "sequential_refinement": [
        "First prompt in sequence",
        "Second prompt in sequence"
      ]
    }
  ]
}
```

3. **Run your custom test**

```bash
python -m scripts.run_all_tests --test-suite test_suites/my_custom_test.json
```

### Analyzing Results

#### Using the Jupyter Notebooks

The `examples` directory contains Jupyter notebooks to help you analyze results:

1. Open the results analysis notebook:

```bash
jupyter notebook examples/results_analysis.ipynb
```

2. Update the path to your results directory:

```python
results_dir = "path/to/your/results"
```

3. Run the notebook cells to generate visualizations and insights.

#### Custom Analysis

You can also create your own analysis scripts using the framework's analysis modules:

```python
from llm_tester.analyzers import response_classifier
from llm_tester.utils import data_processing

# Load results
results = data_processing.load_results("path/to/results")

# Analyze patterns
classifier = response_classifier.ResponseClassifier()
analyzed_results = [classifier.classify(result["response"]) for result in results]

# Process as needed
```

### Next Steps

- Explore more test suites in the `test_suites` directory
- Check out the example notebooks in the `examples` directory
- Read the full documentation in the `README.md` file

For more detailed information, see the full [Implementation Guide](docs/implementation_guide.md).