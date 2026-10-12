"""Bulwark inside Claude Code: hooks that scan untrusted tool results and gate risky follow-up actions.

    PostToolUse       scan what WebFetch, WebSearch, Read, Bash, Grep and MCP tools brought in; warn Claude on
                      a hit and record taint for the session
    PreToolUse        taint and provenance gating: network commands, fetches and "send" tools after untrusted
                      content, and destinations that only appeared in that content -> ask or deny
    UserPromptSubmit  remember the destinations the user named (they are trusted)
    SessionEnd        delete the session's state file

See `bulwark.claude_code.hooks` for the stdin/stdout contract and the README section "Use with Claude Code".
"""
