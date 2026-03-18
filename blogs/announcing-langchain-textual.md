# Announcing langchain-textual: PII Redaction and Synthesis for LangChain

Every organization building with AI faces the same tension: the data that makes models useful is the same data that regulations say you can't expose. Customer support tickets contain names and account numbers. Clinical notes are full of PHI. Chat logs have emails, phone numbers, and addresses woven through every conversation.

This isn't a niche problem. It shows up everywhere sensitive data meets AI:

- **Training and fine-tuning LLMs** on real-world data — except that data has patient names, SSNs, and credit card numbers baked in.
- **RAG pipelines** where documents get chunked, embedded, and stored in vector databases. PII persists at every layer — in the source documents, the chunks, the embeddings, and the retrieval results.
- **LLM proxies** that sit between users and models, where you need to scrub PII from prompts before they leave your infrastructure and from completions before they reach the user.
- **Agent workflows** where tools pull data from databases, APIs, and files — each step potentially surfacing and forwarding PII to the next.
- **Compliance workflows** — HIPAA de-identification (both Safe Harbor and Expert Determination), GDPR right-to-erasure, PCI cardholder data rules.
- **Test and lower environments** that need realistic data without the regulatory exposure of production PII.

Regex and rule-based approaches can catch a phone number that looks like `(555) 123-4567`, but they fall apart on context-dependent PII. Is "Jordan" a person's name or a country? Is "April" a name or a month? Is "1600 Pennsylvania Avenue" a location that needs redaction or a well-known reference? You need NER models that understand context, and you need them to work across text, JSON, HTML, PDFs, images, and CSVs — not just plain strings.

That's what [Tonic Textual](https://tonic.ai/textual) does. And now, with `langchain-textual`, those capabilities are available as standard LangChain tools.

## What Tonic Textual Does

Tonic Textual is a PII detection and transformation engine. At its core is a named entity recognition (NER) system that identifies 46+ entity types across 50+ languages — names, addresses, phone numbers, emails, SSNs, credit cards, medical record numbers, and more.

What makes it different from basic NER libraries is what happens after detection. Textual offers two modes:

**Redaction** replaces PII with labeled placeholders:

```
Input:  "Contact John Smith at john@example.com"
Output: "Contact [NAME_GIVEN_a1b2] [NAME_FAMILY_c3d4] at [EMAIL_ADDRESS_e5f6]"
```

The placeholders are tagged with their entity type and a consistent identifier, so you can track which replacements correspond to the same original entity across a document.

**Synthesis** replaces PII with realistic fake data that preserves the structure and statistical properties of the original:

```
Input:  "Contact John Smith at john@example.com"
Output: "Contact Maria Chen at maria.chen@gmail.com"
```

Synthesized data keeps downstream analytics, model training, and testing valid. A synthesized email address is still a valid email. A synthesized phone number has the right format. A synthesized name is still a plausible name. This matters when you're training models on de-identified data — placeholder tokens like `[NAME_GIVEN_xxxx]` distort the distribution your model learns from, while synthesized replacements preserve it.

Textual handles all of this across multiple formats: plain text, JSON (with structural awareness of keys vs. values), HTML (preserving markup), PDFs, images (OCR + redaction), and tabular data (CSV, TSV). It runs in the cloud or self-hosted on your infrastructure, with enterprise controls like RBAC, SSO, and audit logging.

## What langchain-textual Adds

`langchain-textual` wraps Tonic Textual's API into five LangChain tools that any chain or agent can use:

| Tool | Input | Use for |
|------|-------|---------|
| `TonicTextualRedactText` | Plain text string | Raw text, `.txt` file contents |
| `TonicTextualRedactJson` | JSON string | Raw JSON, `.json` file contents |
| `TonicTextualRedactHtml` | HTML string | Raw HTML, `.html`/`.htm` file contents |
| `TonicTextualRedactFile` | File path | PDFs, images (JPG, PNG), CSVs, TSVs |
| `TonicTextualPiiTypes` | None | List all 46+ supported PII entity types |

These are standard LangChain `BaseTool` subclasses. You can pass them to any agent, chain, or tool-calling LLM. The installation is one line:

```bash
pip install langchain-textual
```

## Architecture Deep Dive

### Shared Base Class

All redaction tools inherit from a shared `_BaseTonicTextual` base class that handles client initialization and configuration:

```python
class _BaseTonicTextual(BaseTool):
    client: TextualNer = Field(default=None)
    tonic_textual_api_key: SecretStr = Field(default=SecretStr(""))
    tonic_textual_base_url: str | None = None
    generator_default: Literal["Off", "Redaction", "Synthesis"] | None = None
    generator_config: dict[str, Literal["Off", "Redaction", "Synthesis"]] = Field(
        default_factory=dict
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        return initialize_client(values)

    def _build_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.generator_default is not None:
            kwargs["generator_default"] = self.generator_default
        if self.generator_config:
            kwargs["generator_config"] = self.generator_config
        return kwargs
```

The `model_validator` runs before instantiation, reading the API key from the constructor argument or the `TONIC_TEXTUAL_API_KEY` environment variable and initializing the Textual client. Each tool subclass then implements `_run` with a single call to the appropriate Textual API method — `client.redact()` for text, `client.redact_json()` for JSON, `client.redact_html()` for HTML.

The `_build_kwargs()` method centralizes the configuration options that all Textual API methods accept — `generator_default` and `generator_config` — so each tool's `_run` method stays focused on its format-specific logic.

### Guiding the Agent with `args_schema`

When an LLM agent has multiple tools available, it decides which one to call based on the tool's `name`, `description`, and input schema. LangChain generates the input schema from a Pydantic model — and crucially, the `Field(description=...)` values are included in the schema the LLM sees.

We define explicit input schemas for each tool:

```python
class _RedactTextInput(BaseModel):
    text: str = Field(
        description=(
            "Plain text that may contain PII. "
            "For .txt files, read the file first and pass the contents here."
        )
    )

class _RedactFileInput(BaseModel):
    file_path: str = Field(
        description=(
            "Absolute path to the file to redact. "
            "Supported types: JPG, PNG, PDF, CSV, TSV. "
            "Do NOT use for .txt, .json, .html, or .htm files — use the "
            "dedicated text, JSON, or HTML redaction tools instead."
        )
    )
    output_path: str | None = Field(
        default=None,
        description=(
            "Path to write the redacted file. "
            "Defaults to <original_name>_redacted.<ext> in the same directory."
        ),
    )
```

These descriptions aren't just documentation — they're instructions that the LLM reads when deciding how to call the tool. By stating what each parameter expects and explicitly noting what *not* to use each tool for, we make it much more likely the agent picks the right tool on the first try.

### Self-Correcting Error Messages

Even with good descriptions, agents sometimes pick the wrong tool. Rather than returning a raw Python traceback, each tool validates its input and returns actionable error messages that tell the agent exactly what to do instead:

```python
# In TonicTextualRedactText._run():
try:
    json.loads(text)
    return (
        "Error: input looks like JSON, not plain text. "
        "Use tonic_textual_redact_json for .json files (read contents first)."
    )
except (json.JSONDecodeError, TypeError):
    pass
```

```python
# In TonicTextualRedactFile._run():
if ext_lower == ".txt":
    return (
        "Error: .txt files are not supported by this tool. "
        "Use tonic_textual_redact for .txt files (read contents first)."
    )
```

When the text tool receives JSON, it doesn't just fail — it tells the agent "this looks like JSON, use the JSON tool instead." When the file tool receives a `.html` path, it redirects to the HTML tool. The agent can self-correct in the next step without human intervention.

This pattern works because LLM agents treat tool outputs as observations in their reasoning loop. An error message that says `ConnectionError('...')` gives the agent nothing to work with. An error message that says "use tonic_textual_redact_json instead" gives it a clear next action.

### The PII Types Tool

The `TonicTextualPiiTypes` tool is deliberately simple — it reads the `PiiType` enum from the Tonic Textual SDK and returns all 46 supported entity type names:

```python
class TonicTextualPiiTypes(BaseTool):
    name: str = "tonic_textual_pii_types"
    description: str = (
        "Lists all PII entity types supported by Tonic Textual. "
        "Call this tool to discover valid entity type names (e.g. "
        "NAME_GIVEN, EMAIL_ADDRESS, CREDIT_CARD) that can be used in "
        "generator_config to control per-type redaction behavior. "
        "No input is required."
    )

    def _run(self, query: str = "", **kwargs) -> str:
        return ", ".join(member.value for member in PiiType)
```

No API key needed, no network call. This tool exists so an agent (or a developer in a REPL) can discover valid entity type names for use in `generator_config` without having to look up the documentation.

## Walkthrough: Building a PII-Redacting Agent

Here's a complete example of a LangChain ReAct agent that can redact text and files:

```python
from langchain_textual import (
    TonicTextualRedactText,
    TonicTextualRedactFile,
    TonicTextualPiiTypes,
)
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [
    TonicTextualRedactText(),
    TonicTextualRedactFile(),
    TonicTextualPiiTypes(),
]
agent = create_react_agent(llm, tools)
```

With two environment variables set (`TONIC_TEXTUAL_API_KEY` and `OPENAI_API_KEY`), that's a working agent. Let's trace what happens with a few prompts.

### Redacting text

**Prompt:** "Redact this: My name is John Smith and my email is john@example.com"

The agent sees three tools. Based on the descriptions and the fact that the input is plain text, it calls `tonic_textual_redact` with:

```json
{"text": "My name is John Smith and my email is john@example.com"}
```

Textual's NER model identifies `John Smith` as `NAME_GIVEN` + `NAME_FAMILY` and `john@example.com` as `EMAIL_ADDRESS`. The tool returns:

```
My name is [NAME_GIVEN_a1b2] [NAME_FAMILY_c3d4] and my email is [EMAIL_ADDRESS_e5f6].
```

The agent passes this back to the user.

### Redacting a PDF

**Prompt:** "Redact the file /tmp/medical_record.pdf"

The agent recognizes this as a file path and calls `tonic_textual_redact_file` with:

```json
{"file_path": "/tmp/medical_record.pdf"}
```

Behind the scenes, the tool:
1. Opens the file in binary mode
2. Uploads it to Tonic Textual via `client.start_file_redaction()`
3. Polls for completion via `client.download_redacted_file()`
4. Writes the redacted PDF to `/tmp/medical_record_redacted.pdf`
5. Returns the output path

The agent tells the user: "The redacted file has been saved to /tmp/medical_record_redacted.pdf."

### Wrong tool, self-correction

**Prompt:** "Redact this JSON: {\"name\": \"Jane Doe\", \"ssn\": \"123-45-6789\"}"

The agent might initially call the text tool. The text tool detects valid JSON in the input and returns:

```
Error: input looks like JSON, not plain text. Use tonic_textual_redact_json for .json files (read contents first).
```

The agent reads this, calls the JSON tool instead, and gets the correctly redacted JSON. The user never sees the intermediate error — they just get the right result.

## Synthesis vs. Redaction: Choosing the Right Mode

The default mode is redaction — PII is replaced with placeholders like `[EMAIL_ADDRESS_xxxx]`. This is the safest option when you need to guarantee that no real PII appears in the output.

But placeholders have a cost. If you're training a language model on de-identified clinical notes, every `[NAME_GIVEN_xxxx]` is a token the model learns to predict that has no relationship to how real names appear in text. The model's understanding of sentence structure around names gets degraded.

Synthesis mode solves this by generating realistic replacements:

```python
tool = TonicTextualRedactText(generator_default="Synthesis")
tool.invoke("Patient John Smith, DOB 03/15/1982, was admitted on 01/10/2024.")
# "Patient Maria Chen, DOB 07/22/1975, was admitted on 04/03/2024."
```

The synthesized data preserves the format and distribution of the original — names are still names, dates are still valid dates, SSNs have the right digit pattern. Downstream models and analytics work correctly on the transformed data.

### Per-Entity Control with `generator_config`

Often you want different handling for different entity types. Names might be safe to synthesize, but SSNs should always be hard-redacted. `generator_config` gives you this control:

```python
tool = TonicTextualRedactText(
    generator_default="Off",          # ignore everything by default
    generator_config={
        "NAME_GIVEN": "Synthesis",     # replace with fake first names
        "NAME_FAMILY": "Synthesis",    # replace with fake last names
        "EMAIL_ADDRESS": "Redaction",  # hard-redact emails
        "US_SSN": "Redaction",         # hard-redact SSNs
    },
)
```

This configuration says: synthesize names so the text reads naturally, hard-redact emails and SSNs for maximum safety, and leave everything else untouched. The `generator_default` of `"Off"` means any entity type not listed in `generator_config` passes through unchanged.

This composability matters for compliance. HIPAA's Safe Harbor method requires removal or generalization of 18 specific identifier categories. With `generator_config`, you can map each of those categories to the appropriate handling — redaction for direct identifiers like SSNs, synthesis for quasi-identifiers like dates where you need to preserve analytical utility.

Not sure which entity types are available? The `TonicTextualPiiTypes` tool returns all 46:

```python
from langchain_textual import TonicTextualPiiTypes
TonicTextualPiiTypes().invoke("")
# "NUMERIC_VALUE, LANGUAGE, MONEY, ..., US_SSN, CREDIT_CARD, EMAIL_ADDRESS, ..."
```

## Under the Hood: How Tonic Textual Detects PII

Tonic Textual uses a transformer-based NER model, not regex patterns. This is the difference between matching `\d{3}-\d{2}-\d{4}` (which catches anything that looks like an SSN, including dates and product codes) and understanding from context that "her social is 123-45-6789" contains an SSN while "case number 123-45-6789" does not.

The model detects 46 entity types out of the box:

- **Personal identifiers:** `NAME_GIVEN`, `NAME_FAMILY`, `PERSON`, `DOB`, `PERSON_AGE`, `GENDER_IDENTIFIER`, `US_SSN`, `US_PASSPORT`, `US_DRIVER_LICENSE`
- **Contact information:** `EMAIL_ADDRESS`, `PHONE_NUMBER`, `URL`, `USERNAME`, `PASSWORD`
- **Financial:** `CREDIT_CARD`, `CC_EXP`, `CVV`, `US_BANK_NUMBER`, `IBAN_CODE`, `US_ROUTING_TRANSIT_NUMBER`
- **Location:** `LOCATION`, `LOCATION_ADDRESS`, `LOCATION_CITY`, `LOCATION_STATE`, `LOCATION_ZIP`, `LOCATION_COUNTRY`, `LOCATION_COMPLETE_ADDRESS`
- **Healthcare:** `HEALTHCARE_ID`, `MEDICAL_LICENSE`
- **Temporal:** `DATE_TIME`
- **Other:** `ORGANIZATION`, `OCCUPATION`, `IP_ADDRESS`, `NUMERIC_VALUE`, `NUMERIC_PII`, and more

Each detection comes with a confidence score, and the API returns the full list of detected entities alongside the redacted text — so you can audit what was found and how it was handled.

For formats like JSON and HTML, Textual applies structural awareness. In JSON, it understands the difference between keys (which are typically not PII) and values (which often are). In HTML, it preserves the markup structure while redacting only the text content. This means `<a href="mailto:john@example.com">John Smith</a>` gets the email and name redacted while the `<a>` tag structure is preserved.

## What's Next

This initial release covers the core redaction use cases — text, JSON, HTML, and binary files. We're continuing to expand format support and add new capabilities.

The package is open source under the MIT license. You can find it at:

- **PyPI:** [langchain-textual](https://pypi.org/project/langchain-textual/)
- **GitHub:** [tonicai/langchain-tonic-textual](https://github.com/tonicai/langchain-tonic-textual)
- **Examples:** a self-contained agent example lives in the `examples/` directory of the repo

To get started:

```bash
pip install langchain-textual
export TONIC_TEXTUAL_API_KEY="your-api-key"
```

```python
from langchain_textual import TonicTextualRedactText

tool = TonicTextualRedactText()
tool.invoke("My name is John Smith and my email is john@example.com.")
```

For the full Tonic Textual platform — including the web UI, dataset management, custom entity training, and enterprise deployment options — visit [tonic.ai/textual](https://tonic.ai/textual).
