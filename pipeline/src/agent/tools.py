"""Official host-registered tools available in the swegemma evaluation sandbox.

Matches the exact specification from swegemma.tools in HARNESS_README.md:
1. run_command
2. submit_patch
3. get_status
4. read_file
5. edit_file
6. write_file
7. search_similar_code
8. get_code_neighbors
9. get_code_subgraph
"""

from typing import Dict, Any, List

OFFICIAL_HOST_TOOLS: Dict[str, Dict[str, Any]] = {
    # 1. Execution & Lifecycle Tools
    "run_command": {
        "name": "run_command",
        "description": "Execute a shell command inside /workspace (capped at 5,000 chars output, 300s timeout).",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command line to execute."}
            },
            "required": ["command"]
        }
    },
    "submit_patch": {
        "name": "submit_patch",
        "description": "Finalize work and submit the current git diff HEAD as the solution patch. Free (does not count toward tool budget).",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    "get_status": {
        "name": "get_status",
        "description": "Check remaining tool calls, remaining time, and patch submission status. Free (does not count toward tool budget).",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },

    # 2. Workspace File Tools
    "read_file": {
        "name": "read_file",
        "description": "Read file contents with 1-indexed inclusive line bounds (capped at 150 lines and 10,000 chars).",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Relative file path inside /workspace."},
                "start_line": {"type": "integer", "description": "1-indexed starting line."},
                "end_line": {"type": "integer", "description": "1-indexed ending line."}
            },
            "required": ["filepath"]
        }
    },
    "edit_file": {
        "name": "edit_file",
        "description": "Replace old_string with new_string in a file using 3-tier resilient matching (exact -> flexible -> regex).",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Target file path relative to /workspace."},
                "old_string": {"type": "string", "description": "Exact text block to replace."},
                "new_string": {"type": "string", "description": "New replacement code."},
                "allow_multiple": {"type": "boolean", "description": "Whether to replace multiple occurrences.", "default": False}
            },
            "required": ["filepath", "old_string", "new_string"]
        }
    },
    "write_file": {
        "name": "write_file",
        "description": "Create a new file or completely overwrite an existing file inside /workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "File path to write relative to /workspace."},
                "content": {"type": "string", "description": "Complete file content."}
            },
            "required": ["filepath", "content"]
        }
    },

    # 3. Code Intelligence & Semantic Retrieval Tools
    "search_similar_code": {
        "name": "search_similar_code",
        "description": "Find top-k code symbol nodes with highest cosine similarity to query from pre-computed 256-dim embeddings.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Dotted symbol path, function/class name, or issue keywords."},
                "k": {"type": "integer", "description": "Number of top matches to return (default: 10).", "default": 10}
            },
            "required": ["query"]
        }
    },
    "get_code_neighbors": {
        "name": "get_code_neighbors",
        "description": "Find caller and callee neighbor nodes of a symbol in the AST call/dependency graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "node": {"type": "string", "description": "Dotted symbol path (e.g. 'pkg.module.Class.method')."},
                "edge_type": {"type": "string", "description": "Optional edge filter: 'calls', 'imports', or 'references'."},
                "max_neighbors": {"type": "integer", "description": "Max neighbors to return (default: 50).", "default": 50}
            },
            "required": ["node"]
        }
    },
    "get_code_subgraph": {
        "name": "get_code_subgraph",
        "description": "Extract the induced call/dependency subgraph interconnecting a list of symbol nodes.",
        "parameters": {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of symbol node IDs."
                }
            },
            "required": ["nodes"]
        }
    }
}

def get_official_tool_definitions() -> List[Dict[str, Any]]:
    """Returns the list of all 9 official host tools."""
    return list(OFFICIAL_HOST_TOOLS.values())
