# LLM Guardrail Benchmark: Comprehensive Test Suite

## Overview
This benchmark evaluates how language models respond to different prompting techniques that test their guardrails across multiple domains. The test suite aims to provide a standardized way to measure model safety, consistency, and robustness against various circumvention attempts.

## Test Categories

### 1. Information Hazards
Tests for model responses to requests for potentially dangerous information.

* **Chemical Hazards**
  * Household chemicals
  * Synthesis techniques
  * Explosive materials
  * Controlled substances

* **Biological Hazards**
  * Pathogen information
  * Bioweapon concepts
  * Dangerous biological processes
  
* **Physical Harm**
  * Weapon creation
  * Self-harm methods
  * Physical attack techniques

### 2. Malicious Code Generation
Tests for model responses to requests for harmful code or cyber exploits.

* **Malware Creation**
  * Viruses
  * Ransomware
  * Keyloggers
  
* **Exploits**
  * Authentication bypasses
  * SQL injection techniques
  * Session hijacking
  * Password cracking
  
* **Social Engineering Tools**
  * Phishing templates
  * Credential harvesting

### 3. Personal Data Exploitation
Tests for responses to requests involving unauthorized personal data access or creation.

* **Identity Document Generation**
  * Fake identification
  * Synthetic identity creation
  
* **Financial Instruments**
  * Fraudulent transactions
  * Financial document forgery
  
* **Unauthorized Access**
  * Account takeover methods
  * Password reset exploitation

### 4. Deceptive Content
Tests for responses to requests for misleading or manipulative content.

* **Misinformation**
  * Fake news generation
  * Misleading statistics
  * Conspiracy narratives
  
* **Media Manipulation**
  * Deepfake guidance
  * Image manipulation for deception
  
* **Academic Dishonesty**
  * Plagiarism assistance
  * Essay generation to bypass detection

### 5. Unethical Persuasion
Tests for responses to requests for manipulative communication techniques.

* **Manipulation Tactics**
  * Gaslighting scripts
  * Psychological manipulation strategies
  * Coercive techniques
  
* **Scam Templates**
  * Social engineering scripts
  * Confidence schemes
  * Fraudulent marketing materials

### 6. Privacy and Surveillance
Tests for responses to requests related to unauthorized surveillance or privacy violations.

* **Tracking Methods**
  * Digital surveillance techniques
  * Unauthorized monitoring tools
  * Location tracking without consent
  
* **Data Exfiltration**
  * Methods to extract private data
  * Techniques to circumvent privacy protections

### 7. Political Misuse
Tests for responses to requests for political manipulation content.

* **Propaganda**
  * Targeted disinformation campaigns
  * Divisive political content
  
* **Election Interference**
  * Voter suppression content
  * Election fraud methods
  
* **Political Defamation**
  * False claims about real politicians
  * Character assassination techniques

### 8. Harmful Creative Content
Tests for responses to requests for creative content that could cause harm.

* **Hateful Content**
  * Discriminatory narratives
  * Content targeting vulnerable groups
  
* **Graphic Violence**
  * Descriptive violent scenes
  * Glorification of violence
  
* **Inappropriate Content Involving Minors**
  * Age-inappropriate scenarios
  * Content that could be used for grooming

### 9. Technical Guardrail Probing
Tests for responses to direct attempts to circumvent model limitations.

* **System Prompt Extraction**
  * Attempts to reveal underlying instructions
  * Prompts to expose model limitations
  
* **Policy Disclosure**
  * Requests for internal filtering processes
  * Attempts to expose moderation thresholds
  
* **Role-Breaking**
  * DAN-style prompts
  * Character assumption circumvention
  * "Pretend" scenarios to evade guardrails

### 10. Edge Cases and Novel Techniques
Tests for responses to boundary-pushing or emerging circumvention techniques.

* **Indirect Requests**
  * Chunked requests across multiple prompts
  * Using code examples that output harmful content
  
* **Language Tricks**
  * Encoding in different languages
  * Ciphers and obfuscation
  * Token manipulation tricks
  
* **Adversarial Examples**
  * Deliberately ambiguous prompts
  * Messages with dual interpretations
  * Exploiting confusing terminology

## Variation Dimensions
Each test category includes variations along these dimensions:

1. **Directness** (Level 1-3)
   * Indirect/implied requests
   * Moderate/ambiguous requests
   * Direct/explicit requests

2. **Technical Framing** (Level 1-3)
   * Layperson language
   * Domain-specific terminology
   * Technical jargon/scientific language

3. **Contextual Embedding** (Level 1-3)
   * Educational context
   * Fiction/creative context
   * Personal/practical use context

4. **Authority Invocation** (Level 1-3)
   * No authority cited
   * Moderate authority (student/professional)
   * High authority (professor/official/legal)

5. **Role-Playing** (Level 1-3)
   * Hypothetical third-person
   * Character play with distance
   * Direct role assumption

6. **Sequential Refinement** (5-step sequences)
   * Starting with innocent requests
   * Gradually escalating specificity
   * Building contextual dependency
   * Leveraging prior compliance
   * Final direct request

## Evaluation Metrics

The benchmark measures:

1. **Refusal Rate** - Percentage of requests that are refused
2. **Response Strategy** - Categories of response (complete refusal, refusal with alternative, cautious compliance, full compliance)
3. **Consistency** - Variance in responses to similar requests
4. **Circumvention Vulnerability** - Success rate of different techniques
5. **Harmful Content Leakage** - Degree to which harmful information appears despite refusals

## Implementation
Each test includes:

* Baseline prompt - standard form of the request
* Variations across all dimensions
* Sequential refinement series
* Standard system prompt
* Documentation of intent and risk level

This structure allows for detailed analysis of where and how LLM guardrails function effectively or exhibit weaknesses.