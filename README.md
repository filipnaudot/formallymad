<div align='center'>
    <picture>
        <source media="(prefers-color-scheme: light)" srcset="./docs/lightmode-logo-formallymad.png">
        <img alt="formallymad logo" src="./docs/darkmode-logo-formallymad.png" width="50%" height="50%">
    </picture>
</div>


# Formally MAD - Multi-Agent RecSys Framework

Start by cloning the **GitHub** repository, then create and activate a virtual environment before installing with `pip`:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

Or with visualizer support:

```bash
pip install -e .[visualize]
```

Add your OpenAI API key to a `.env` file at the project root:

```
OPENAI_API_KEY=sk-...
```

Then run:

```bash
python main.py
```


## Tools

Agents have access to a set of tools defined in `src/formallymad/tools.py`. To add a new tool, implement it as a regular Python function with a docstring (used as the tool description) and typed parameters, then register it in `TOOL_REGISTRY` at the bottom of the same file:

```python
def my_tool(param: str) -> dict:
    """Does something useful given a param."""
    ...
    return {"result": ...}

TOOL_REGISTRY = {
    ...,
    "my_tool": my_tool,
}
```

The framework will pick up the function signature automatically.
