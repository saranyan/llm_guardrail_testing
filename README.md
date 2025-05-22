# LLM Guardrail Testing Framework

A systematic framework for testing how different language models respond to variations in prompt phrasing and evaluating the effectiveness of their safety guardrails.

## Project Overview

This framework enables researchers and developers to:

1. Test how variations in language affect LLM responses
2. Measure the effectiveness of safety guardrails across different models
3. Identify patterns in how models handle boundary-pushing requests
4. Compare responses between different LLMs (currently supporting Claude and OpenAI models)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/saranyan/llm_guardrail_testing.git
cd llm_guardrail_testing

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys
echo "ANTHROPIC_API_KEY=your_anthropic_api_key" > .env
echo "OPENAI_API_KEY=your_openai_api_key" >> .env

# Run a test suite
python -m scripts.run_all_tests --test-suite test_suites/information_boundaries.json

# Generate a report
python -m scripts.generate_report --input-dir results/processed --output results/reports/report.html
```

## 📁 Project Structure

```
llm-guardrail-testing/
├── src/           # Code to run the tests
│   ├── guardrail_benchmark.py           # Core functionality
│   ├── generate_html_from_run.py        # generates HTML file from run for viz
|   ├── console_summary_from_run.py      # generates console report from run
│
├── scripts/              # Utility scripts
├── test_suites/          # Test definitions
├── config/               # Configuration files
├── results/              # Results directory (created at runtime)
├── examples/             # Example code and notebooks
└── tests/                # Unit tests
```

##  Test Suites

The framework includes several pre-defined test suites:

- **Information Boundaries**: Tests requests for potentially dangerous information
- **Creative Content**: Tests requests for inappropriate creative content
- **Code Generation**: Tests requests for harmful code
- **Policy Extraction**: Tests attempts to extract model policies
- **Personal Data**: Tests generation of sensitive personal information
- **Harmful Advice**: Tests requests for harmful advice
- **Role Circumvention**: Tests attempts to circumvent the model's role

Each test suite examines different dimensions of variation:

- **Directness**: From indirect to explicit requests
- **Technical Framing**: From layperson to technical language
- **Contextual Embedding**: Different contexts for the same request
- **Sequential Refinement**: Building up to problematic requests gradually

##  Analysis Features

The framework automatically analyzes responses for:

- Response classification (refusal, cautious compliance, full compliance)
- Presence of disclaimers, refusal language, or alternatives
- Response length and other metrics
- Comparison across models and prompt variations

##  Example Use Cases

1. **Research**: Study how linguistic variations affect AI safety boundaries
2. **Development**: Test your own models' guardrails during development
3. **Evaluation**: Compare safety measures across different commercial LLMs
4. **Education**: Demonstrate AI safety challenges and solutions

##  Documentation

- [Getting Started Guide](docs/getting_started.md)
- [Implementation Guide](docs/implementation_guide.md)
- [API Reference](docs/api_reference.md)
- [Test Suite Format](docs/test_suite_format.md)

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Ethical Considerations

This framework is designed for research and educational purposes to understand and improve AI safety mechanisms. Please use responsibly:

- Do not use this tool to deliberately circumvent safety measures for harmful purposes
- Consider reporting any unexpected vulnerabilities discovered to the relevant AI providers
- Follow responsible disclosure practices for any safety issues identified

## Acknowledgments

- Inspired by research in the field of AI safety and alignment