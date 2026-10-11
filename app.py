from flask import Flask, request, render_template, jsonify
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
import json
import re
import subprocess
import difflib
import logging

# -----------------------------
# App Setup
# -----------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 6 * 1024  # ~6KB request limit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# Utility: Get installed Ollama models
# -----------------------------
def get_installed_models():
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True
        )
        lines = result.stdout.strip().splitlines()
        if len(lines) <= 1:
            return ["phi3"]

        models = []
        for line in lines[1:]:
            if line.strip():
                model_name = line.strip().split()[0]
                base_name = model_name.split(':')[0]
                models.append(base_name)
        return models if models else ["phi3"]

    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logger.warning(f"Ollama CLI issue: {e}")
        return ["phi3"]
    except Exception as e:
        logger.exception("Unexpected error in get_installed_models")
        return ["phi3"]


# -----------------------------
# Utility: Generate HTML diff
# -----------------------------
def generate_diff_html(original: str, edited: str) -> str:
    """Generate HTML with <del> and <ins> for word-level differences."""
    orig_words = original.split()
    edit_words = edited.split()
    matcher = difflib.SequenceMatcher(None, orig_words, edit_words)
    result = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            result.extend(orig_words[i1:i2])
        elif tag == 'delete':
            for word in orig_words[i1:i2]:
                result.append(f'<del>{word}</del>')
        elif tag == 'insert':
            for word in edit_words[j1:j2]:
                result.append(f'<ins>{word}</ins>')
        elif tag == 'replace':
            for word in orig_words[i1:i2]:
                result.append(f'<del>{word}</del>')
            for word in edit_words[j1:j2]:
                result.append(f'<ins>{word}</ins>')

    return ' '.join(result)


# -----------------------------
# Utility: Parse LLM JSON response
# -----------------------------
def parse_llm_json_response(output: str) -> str:
    """Extract edited_text from LLM output, ignoring extra commentary."""
    # Look for JSON-like structure containing "edited_text"
    match = re.search(r'\{[^{}]*"edited_text"[^{}]*\}', output, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return str(parsed.get("edited_text", "")).strip()
        except (json.JSONDecodeError, TypeError):
            pass
    return output.strip()


# -----------------------------
# Routes
# -----------------------------

@app.route('/')
def index():
    models = get_installed_models()
    return render_template('index.html', models=models)


@app.route('/review', methods=['POST'])
def review():
    data = request.get_json(silent=True) or {}

    # Validate input
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    if len(text) > 5000:
        return jsonify({'error': 'Text too long (max 5000 characters)'}), 400

    types = data.get('improvementTypes') or ['grammar', 'clarity', 'style']
    if not isinstance(types, list):
        types = [str(types)]
    types = [t.strip() for t in types if t.strip()]
    if not types:
        types = ['grammar', 'clarity', 'style']
    types_str = ', '.join(types)

    # Select model
    model_name = data.get('model')
    installed_models = get_installed_models()
    if not model_name or model_name not in installed_models:
        model_name = installed_models[0]

    # Initialize LLM
    try:
        llm = ChatOllama(
            model=model_name,
            temperature=0.3,
            num_ctx=4096
        )
    except Exception as e:
        logger.error(f"Failed to initialize ChatOllama: {e}")
        return jsonify({'error': 'Model initialization failed'}), 500

    # Build strict prompt
    prompt_template = (
        "You are an expert editor. Improve the following text for: {types}.\n"
        "Fix grammar, punctuation, clarity, and style. Preserve all original meaning.\n"
        "Return ONLY a valid JSON object with one key: 'edited_text'.\n"
        "Do NOT include any other text, explanation, or commentary.\n"
        "Example: {{\"edited_text\": \"Corrected sentence.\"}}\n\n"
        "Text:\n{text}"
    )
    prompt = PromptTemplate(input_variables=['text', 'types'], template=prompt_template)
    formatted_prompt = prompt.format(text=text, types=types_str)

    # Call LLM
    try:
        logger.info(f"Calling model '{model_name}' with {len(text)} chars")
        response = llm.invoke(formatted_prompt)
        raw_output = response.content if hasattr(response, 'content') else str(response)
        edited_text = parse_llm_json_response(raw_output)

        # Generate diff
        diff_html = generate_diff_html(text, edited_text)

    except Exception as e:
        logger.exception("LLM processing failed")
        return jsonify({'error': 'Text improvement failed', 'details': str(e)}), 500

    return jsonify({
        'original_text': text,
        'edited_text': edited_text,
        'diff_html': diff_html,
        'model_used': model_name
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'models_available': get_installed_models()
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)