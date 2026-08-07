GET_SUMMARY_BY_TITLE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_summary_by_title",
        "description": (
            "Returns the complete and detailed summary of a book. "
            "Use this function after selecting exactly one book "
            "from the retrieved RAG context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": (
                        "The exact title copied from the "
                        "retrieved book context."
                    ),
                }
            },
            "required": [
                "title",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}