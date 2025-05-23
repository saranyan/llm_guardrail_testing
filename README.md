# LLM Guardrail Testing Framework

A systematic framework for testing how different language models respond to variations in prompt phrasing and evaluating the effectiveness of their safety guardrails. Check * [Details here ](https://www.saranyan.com/research/safety/llm-guardrail-benchmark)

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
├── src/                                 # Code to run the tests
│   ├── guardrail_benchmark.py           # Core functionality
│   ├── generate_html_from_run.py        # generates HTML file from run for viz
|   ├── console_summary_from_run.py      # generates console report from run
|   ├── config.json
├── .env file                             #create this with your API keys
├── benchmark_data/                       # benchmark data in json format
├── sample_run_results/                   # Sample generated data for reference

```

## Test Suites

The framework includes comprehensive test suites covering critical AI safety domains:

### Core Test Categories

- **Harmful Misinformation** - Tests resistance to dangerous health advice, unproven medical claims, and life-threatening misinformation (turmeric as Alzheimer's cure, fasting to cure autoimmune diseases, raw vegan diets as cancer treatment)

- **Guardrail Evasion** - Evaluates resilience against prompt injection, jailbreak techniques, system manipulation, and progressive instruction escalation designed to bypass safety mechanisms

- **Mental Health Crisis** - Ensures appropriate handling of suicidal ideation, self-harm content, crisis intervention scenarios, and emotional distress across direct and indirect expressions

- **Synthetic Media Abuse** - Tests deepfake generation resistance, celebrity impersonation scripts, revenge content creation, and misinformation disguised as satire

- **Illegal Activity Facilitation** - Validates refusal of drug synthesis instructions, precursor extraction methods, novel designer drug creation, and controlled substance manufacturing

- **Political Manipulation** - Checks protection against voter suppression tactics, election disinformation, polling disruption rumors, and automated suppression systems

### Multi-Dimensional Testing Framework

Each test category examines different vectors of potential circumvention:

- **Directness Levels**: From oblique references to explicit requests (3 escalation levels)
- **Technical Framing**: From layperson language to advanced technical terminology
- **Authority Invocation**: From casual requests to claims of professional/academic authority
- **Role-Playing Scenarios**: From hypothetical discussions to direct role assumption
- **Contextual Embedding**: Educational, fictional, research, and practical use contexts
- **Sequential Refinement**: 5-step conversation flows that gradually escalate from innocent to problematic requests

### Risk Classification

Tests are categorized by risk level:
- **Critical**: Life-threatening content (suicide methods, drug synthesis, medical misinformation)
- **High**: Significant harm potential (deepfakes, voter suppression, jailbreaks)
- **Medium**: Moderate risk scenarios requiring careful handling

Each test includes detailed ethical considerations, expected model behavior, and analysis guidance to ensure responsible evaluation of AI safety mechanisms.

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

## Running this
```bash
#run tests
python guardrail_benchmark.py benchmark_data/sample_test_cases.json
python guardrail_benchmark.py benchmark_data/llm_guardrail_benchmark_full.json

#html page with visualizaitons
python generate_html_from_run.py ./sample_run_results

#console summary from existing run
python console_summary_from_run.py ./sample_run_results  

```
## Sample results

check the sample_run_results folder from my runs and sample_run_results/sample_console_results.txt


##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/adding_to_dataset`)
3. Commit your changes (`git commit -m 'Add more cases to data'`)
4. Push to the branch (`git push origin feature/adding_to_dataset`)
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