"""Boundary tests for the full execution pipeline.

Every scenario is checked at four layers, so that two errors cannot cancel
each other out by only comparing final snapshots:

1. the code parsed out of the Markdown source (SuperFences),
2. the exact code the runner receives and the output it returns,
3. the Markdown tokens sent back to the renderer (fences, placeholders),
4. the final HTML.
"""

from __future__ import annotations

import random
import re
import subprocess
from textwrap import dedent
from typing import TYPE_CHECKING, Any

import pytest
from markdown import Markdown

from markdown_exec import formatter, formatters, validator
from markdown_exec._internal.formatters import bash as bash_module
from markdown_exec._internal.formatters import python as python_module
from markdown_exec._internal.formatters import sh as sh_module
from markdown_exec._internal.rendering import MarkdownConverter, markdown_config

if TYPE_CHECKING:
    from collections.abc import Sequence


class PipelineSpy:
    """Record what crosses each layer of the source-to-HTML pipeline."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Install recorders on the runners and on the Markdown converter."""
        # Layer 1: the code parsed out of the Markdown source.
        self.parsed: list[tuple[str, str]] = []
        # Layer 2: the exact code the runner got and the output it returned.
        self.ran: list[tuple[str, str, str]] = []
        # Layer 3: the Markdown tokens sent back to the renderer.
        self.rendered: list[str] = []

        for module, name, language in (
            (bash_module, "_run_bash", "bash"),
            (sh_module, "_run_sh", "sh"),
            (python_module, "_run_python", "python"),
        ):
            original = getattr(module, name)

            def recording(
                code: str,
                *args: Any,
                original: Any = original,
                language: str = language,
                **kwargs: Any,
            ) -> str:
                try:
                    output = original(code, *args, **kwargs)
                except Exception:
                    self.ran.append((language, code, ""))
                    raise
                self.ran.append((language, code, output))
                return output

            monkeypatch.setattr(module, name, recording)

        original_convert = MarkdownConverter.convert

        def recording_convert(
            converter: MarkdownConverter,
            text: str,
            stash: dict[str, str] | None = None,
            id_prefix: str | None = None,
        ) -> str:
            self.rendered.append(text)
            return original_convert(converter, text, stash=stash, id_prefix=id_prefix)

        monkeypatch.setattr(MarkdownConverter, "convert", recording_convert)

    def format(
        self,
        source: str,
        language: str,
        css_class: str,
        options: dict,
        md: Markdown,
        **kwargs: Any,
    ) -> str:
        """SuperFences `format` callback recording the parsed source."""
        self.parsed.append((language, source))
        return formatter(source, language, css_class, options, md, **kwargs)


def make_markdown(spy: PipelineSpy | None = None) -> Markdown:
    """Build a Markdown instance, optionally wired to a `PipelineSpy`."""
    format_callback = spy.format if spy is not None else formatter
    fences = [
        {
            "name": language,
            "class": language,
            "validator": validator,
            "format": format_callback,
        }
        for language in formatters
    ]
    return Markdown(
        extensions=["pymdownx.superfences", "pymdownx.tabbed", "admonition"],
        extension_configs={"pymdownx.superfences": {"custom_fences": fences}},
    )


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> PipelineSpy:
    """Return a pipeline spy installed for the duration of the test."""
    return PipelineSpy(monkeypatch)


def convert(spy: PipelineSpy, source: str) -> str:
    """Convert a document with a fresh Markdown instance."""
    return make_markdown(spy).convert(source)


# --------------------------------------------------------------------------- #
# Fixed scenarios (at least 12): source -> parsed -> runner -> tokens -> HTML #
# --------------------------------------------------------------------------- #


def test_backtick_fence_python_unicode(spy: PipelineSpy) -> None:
    """Backtick fence with Unicode code and output."""
    source = dedent(
        """
        ```python exec="yes"
        print("héllo → 世界 🐍")
        ```
        """,
    )
    html = convert(spy, source)

    assert spy.parsed == [("python", 'print("héllo → 世界 🐍")')]
    assert spy.ran == [("python", 'print("héllo → 世界 🐍")', "héllo → 世界 🐍\n")]
    assert spy.rendered == ["héllo → 世界 🐍\n"]
    assert html == "<p>héllo → 世界 🐍</p>"


def test_tilde_fence_shell(spy: PipelineSpy) -> None:
    """Tilde fence executes shell like a backtick fence."""
    source = dedent(
        """
        ~~~sh exec="yes"
        echo hi
        ~~~
        """,
    )
    html = convert(spy, source)

    assert spy.parsed == [("sh", "echo hi")]
    assert spy.ran == [("sh", "echo hi", "hi\n")]
    assert spy.rendered == ["hi\n"]
    assert html == "<p>hi</p>"


def test_longer_fence_contains_shorter_fence_in_source(spy: PipelineSpy) -> None:
    """A 4-backtick fence keeps an inner 3-backtick line as code."""
    source = dedent(
        """
        ````python exec="yes" result="text"
        print("```")
        ````
        """,
    )
    html = convert(spy, source)

    assert spy.parsed == [("python", 'print("```")')]
    assert spy.ran == [("python", 'print("```")', "```\n")]
    # Layer 3: the output is safely wrapped in a fence longer than its content.
    assert spy.rendered == ["````````text \n```\n\n````````"]
    assert "<code>" in html
    assert "```" in html


def test_output_containing_fence_is_rendered_as_markdown(spy: PipelineSpy) -> None:
    """Raw output containing a fence is deterministically re-rendered."""
    source = dedent(
        """
        ```python exec="yes"
        print("```")
        ```
        """,
    )
    html = convert(spy, source)

    assert spy.parsed[0][1] == 'print("```")'
    assert spy.ran[0][2] == "```\n"
    assert spy.rendered == ["```\n"]
    assert html == "<p>```</p>"


def test_output_containing_ansi_escapes(spy: PipelineSpy) -> None:
    """ANSI escape sequences survive until the final HTML."""
    source = '```sh exec="yes"\nprintf "\\033[31mred\\033[0m\\n"\n```\n'
    html = convert(spy, source)

    assert spy.parsed == [("sh", 'printf "\\033[31mred\\033[0m\\n"')]
    assert spy.ran == [("sh", 'printf "\\033[31mred\\033[0m\\n"', "\x1b[31mred\x1b[0m\n")]
    assert spy.rendered == ["\x1b[31mred\x1b[0m\n"]
    assert html == "<p>\x1b[31mred\x1b[0m</p>"


def test_very_long_line_without_newlines(spy: PipelineSpy) -> None:
    """A huge single-line output is not wrapped or truncated by default."""
    line = "x" * 5000
    source = f'```python exec="yes"\nprint("{line}", end="")\n```\n'
    html = convert(spy, source)

    assert spy.parsed == [("python", f'print("{line}", end="")')]
    assert spy.ran[0][2] == line
    assert spy.rendered == [line]
    assert html == f"<p>{line}</p>"


def test_output_looking_like_html_is_passed_through(spy: PipelineSpy) -> None:
    """HTML-looking output is injected as-is when not escaped explicitly."""
    source = dedent(
        """
        ```python exec="yes"
        print("<div>hi</div>")
        ```
        """,
    )
    html = convert(spy, source)

    assert spy.parsed == [("python", 'print("<div>hi</div>")')]
    assert spy.ran[0][2] == "<div>hi</div>\n"
    assert spy.rendered == ["<div>hi</div>\n"]
    assert "<div>hi</div>" in html


def test_fence_inside_admonition(spy: PipelineSpy) -> None:
    """A fence nested in an admonition executes and stays inside the admonition."""
    source = dedent(
        """
        !!! note

            ```python exec="yes"
            print(1)
            ```
        """,
    )
    html = convert(spy, source)

    assert spy.parsed == [("python", "print(1)")]
    assert spy.ran == [("python", "print(1)", "1\n")]
    admonition_open = html.index('<div class="admonition note">')
    result = html.index("<p>1</p>")
    admonition_close = html.index("</div>")
    assert admonition_open < result < admonition_close


def test_fence_inside_list_item(spy: PipelineSpy) -> None:
    """A fence nested in a list item executes and stays inside the item."""
    source = dedent(
        """
        - item

            ```python exec="yes"
            print(1)
            ```
        """,
    )
    html = convert(spy, source)

    assert spy.parsed == [("python", "print(1)")]
    assert spy.ran == [("python", "print(1)", "1\n")]
    assert "<li>" in html
    item_open = html.index("<li>")
    result = html.index("<p>1</p>")
    item_close = html.index("</li>")
    assert item_open < result < item_close


def test_fence_inside_tab(spy: PipelineSpy) -> None:
    """A fence nested in a tab executes and stays inside the tab content."""
    source = dedent(
        """
        === "Tab"

            ```python exec="yes"
            print(1)
            ```
        """,
    )
    html = convert(spy, source)

    assert spy.parsed == [("python", "print(1)")]
    assert spy.ran == [("python", "print(1)", "1\n")]
    content_open = html.index('<div class="tabbed-content">')
    result = html.index("<p>1</p>")
    content_close = html.rindex("</div>")
    assert content_open < result < content_close


def test_stdout_stderr_interleaving(spy: PipelineSpy) -> None:
    """Stdout and stderr are merged in chronological order."""
    source = dedent(
        """
        ```sh exec="yes"
        echo out1
        echo err1 >&2
        echo out2
        ```
        """,
    )
    html = convert(spy, source)

    code = "echo out1\necho err1 >&2\necho out2"
    assert spy.parsed == [("sh", code)]
    assert spy.ran == [("sh", code, "out1\nerr1\nout2\n")]
    assert spy.rendered == ["out1\nerr1\nout2\n"]
    assert html == "<p>out1\nerr1\nout2</p>"


def test_crlf_source_is_normalized_before_parsing(spy: PipelineSpy) -> None:
    """CRLF line endings are normalized before reaching parser and runner."""
    source = '```python exec="yes"\r\nprint("cr")\r\n```\r\n'
    html = convert(spy, source)

    assert "\r" not in spy.parsed[0][1]
    assert spy.parsed == [("python", 'print("cr")')]
    assert spy.ran == [("python", 'print("cr")', "cr\n")]
    assert spy.rendered == ["cr\n"]
    assert html == "<p>cr</p>"
    assert "\r" not in html


def test_output_without_trailing_newline(spy: PipelineSpy) -> None:
    """Output without a trailing newline is still wrapped in a paragraph."""
    source = dedent(
        """
        ```sh exec="yes"
        printf nonl
        ```
        """,
    )
    html = convert(spy, source)

    assert spy.parsed == [("sh", "printf nonl")]
    assert spy.ran == [("sh", "printf nonl", "nonl")]
    assert spy.rendered == ["nonl"]
    assert html == "<p>nonl</p>"


def test_session_reuse_within_one_document(spy: PipelineSpy) -> None:
    """Two blocks with the same session name share state in one document."""
    source = dedent(
        """
        ```python exec="1" session="local"
        value = 41
        ```

        ```python exec="1" session="local"
        print(value + 1)
        ```
        """,
    )
    html = convert(spy, source)

    assert [code for _, code in spy.parsed] == ["value = 41", "print(value + 1)"]
    assert len(spy.ran) == 2
    assert spy.ran[1][2] == "42\n"
    assert "42" in html


def test_shell_and_python_sessions_are_separate(spy: PipelineSpy) -> None:
    """Shell and Python blocks never share state, even with the same name."""
    source = dedent(
        """
        ```sh exec="1" session="mix"
        export VALUE=1
        echo "$VALUE"
        ```

        ```python exec="1" session="mix"
        try:
            print(VALUE)
        except NameError:
            print("python-ok")
        ```

        ```sh exec="1" session="mix"
        echo "[$VALUE]"
        ```
        """,
    )
    html = convert(spy, source)

    outputs = [output for _, _, output in spy.ran if output]
    assert outputs == ["1\n", "python-ok\n", "[]\n"]
    assert "python-ok" in html
    assert "<p>[]</p>" in html


def test_same_session_name_does_not_leak_between_documents(spy: PipelineSpy) -> None:
    """Two documents using the same session name do not share state."""
    first_source = dedent(
        """
        ```python exec="1" session="same"
        secret = 42
        ```
        """,
    )
    second_source = dedent(
        """
        ```python exec="1" session="same"
        try:
            print(secret)
        except NameError:
            print("isolated")
        ```
        """,
    )
    first_md = make_markdown(spy)
    second_md = make_markdown(spy)
    first_md.convert(first_source)
    second_html = second_md.convert(second_source)

    assert first_md._markdown_exec_sessions["same"]  # type: ignore[attr-defined]
    assert "secret" not in second_md._markdown_exec_sessions["same"]  # type: ignore[attr-defined]
    assert "isolated" in second_html
    assert "42" not in second_html


@pytest.mark.parametrize(
    ("language", "block"),
    [
        ("sh", "sleep 30"),
        ("bash", "sleep 30"),
    ],
)
def test_shell_timeout_does_not_leak_into_next_block(
    spy: PipelineSpy,
    caplog: pytest.LogCaptureFixture,
    language: str,
    block: str,
) -> None:
    """A timed-out shell process is killed and the next block still runs."""
    source = dedent(
        f"""
        ```{language} exec="1" timeout="0.2" session="late"
        {block}
        ```

        ```{language} exec="1" session="late"
        echo alive
        ```
        """,
    )
    html = convert(spy, source)

    assert spy.parsed[0] == (language, block)
    # The timeout message is rendered as a highlighted code block.
    assert "timed out after 0.2 seconds" in spy.rendered[0]
    assert "timed out" in caplog.text
    # The following block runs in a fresh process and produces its own output.
    assert "<p>alive</p>" in html
    assert any(code == "echo alive" for _, code, _ in spy.ran)


def test_python_timeout_rolls_back_session_state(
    spy: PipelineSpy,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """State set before a Python timeout does not revive in the next block."""
    source = dedent(
        """
        ```python exec="1" session="late-py" timeout="0.2"
        value = 1
        import time
        time.sleep(30)
        ```

        ```python exec="1" session="late-py"
        try:
            print(value)
        except NameError:
            print("state-rolled-back")
        ```
        """,
    )
    html = convert(spy, source)

    assert "timed out after 0.2 seconds" in spy.rendered[0]
    assert "timed out" in caplog.text
    assert "state-rolled-back" in html
    assert spy.ran[1][2] == "state-rolled-back\n"


def test_output_truncation_keeps_original_length(spy: PipelineSpy) -> None:
    """Truncated output gets a marker stating the original total length."""
    source = dedent(
        """
        ```python exec="1" maxoutput="10"
        print("x" * 100)
        ```
        """,
    )
    html = convert(spy, source)

    assert spy.ran[0][2] == "x" * 100 + "\n"
    rendered = spy.rendered[0]
    assert rendered.startswith("x" * 10)
    assert "x" * 11 not in rendered
    assert "showing 10 of 101 characters" in rendered
    assert "showing 10 of 101 characters" in html


def test_html_source_inserts_stash_token(spy: PipelineSpy) -> None:
    """`html` with `source` inserts a placeholder token then restashes it."""
    source = dedent(
        """
        ```python exec="1" html="yes" source="above"
        print("<span>raw</span>")
        ```
        """,
    )
    html = convert(spy, source)

    rendered = spy.rendered[0]
    placeholder = re.search(r'<div class="[0-9a-f-]{36}"></div>', rendered)
    assert placeholder is not None
    # The token never reaches the final HTML; the raw output takes its place.
    assert placeholder.group(0) not in html
    assert "<span>raw</span>" in html
    assert "&lt;span&gt;raw&lt;/span&gt;" in html


# --------------------------------------------------------------------------- #
# Mutations: the layered assertions must pinpoint where the variant deviates  #
# --------------------------------------------------------------------------- #


def test_mutation_one_column_less_indentation(spy: PipelineSpy) -> None:
    """With 3 spaces instead of 4 the fence leaves the list item container."""
    mutated = dedent(
        """
        - item

           ```python exec="yes"
           print(1)
           ```
        """,
    )
    html = convert(spy, mutated)

    # The code itself parses identically...
    assert spy.parsed == [("python", "print(1)")]
    assert spy.ran == [("python", "print(1)", "1\n")]
    # ...but the final layer shows the block is no longer inside the list item.
    assert html.index("</ul>") < html.index("<p>1</p>")


def test_mutation_wrong_closing_fence_length(spy: PipelineSpy) -> None:
    """A closing fence shorter than the opening one does not end the block."""
    mutated = '````python exec="yes"\nprint(1)\n```\nafter\n````\n'
    html = convert(spy, mutated)

    # The trailing text gets swallowed into the executed code.
    assert spy.parsed == [("python", "print(1)\n```\nafter")]
    assert "after" in spy.ran[0][1]
    # The swallowed paragraph never renders as its own block.
    assert "<p>after</p>" not in html


def test_mutation_stderr_not_merged(spy: PipelineSpy, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dropping stderr changes only the runner-output layer, not the parse layer."""

    def unmerged_runner(code: str, **kwargs: Any) -> str:  # noqa: ARG001
        process = subprocess.run(  # noqa: S603
            ["sh", "-c", code],  # noqa: S607
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        spy.ran.append(("sh", code, process.stdout))
        return process.stdout

    monkeypatch.setattr(sh_module, "_run_sh", unmerged_runner)
    source = dedent(
        """
        ```sh exec="yes"
        echo out1
        echo err1 >&2
        echo out2
        ```
        """,
    )
    html = convert(spy, source)

    code = "echo out1\necho err1 >&2\necho out2"
    # Layer 1 is untouched by the runner mutation...
    assert spy.parsed == [("sh", code)]
    # ...layer 2 loses the stderr line...
    assert spy.ran == [("sh", code, "out1\nout2\n")]
    # ...and layer 4 shows exactly that difference.
    assert html == "<p>out1\nout2</p>"
    assert "err1" not in html


# --------------------------------------------------------------------------- #
# Randomized document build order: a second build must be pollution-free      #
# --------------------------------------------------------------------------- #


def _documents() -> list[str]:
    return [
        dedent(
            """
            ```python exec="1" session="build"
            number = 1
            ```

            ```python exec="1" session="build"
            print(f"number is {number}")
            ```
            """,
        ),
        dedent(
            """
            ~~~sh exec="yes"
            echo build-marker
            ~~~
            """,
        ),
        dedent(
            """
            ```python exec="yes"
            print("héllo 🐍")
            ```
            """,
        ),
        dedent(
            """
            ```python exec="1" session="other-doc"
            try:
                print(number)
            except NameError:
                print("not-shared")
            ```
            """,
        ),
        dedent(
            """
            ```python exec="1" maxoutput="8"
            print("y" * 20)
            ```
            """,
        ),
    ]


def _build(order: Sequence[int]) -> dict[int, str]:
    # Emulate what the MkDocs plugin resets between site builds.
    MarkdownConverter.counter = 0
    markdown_config.reset()
    results: dict[int, str] = {}
    documents = _documents()
    for index in order:
        results[index] = make_markdown().convert(documents[index])
    return results


@pytest.mark.parametrize("seed", [0, 17, 4242])
def test_randomized_build_order_is_pollution_free(seed: int) -> None:
    """Two builds in different orders produce identical per-document HTML."""
    rng = random.Random(seed)  # noqa: S311
    order = list(range(len(_documents())))
    second_order = order[:]
    while second_order == order:
        rng.shuffle(second_order)
    rng.shuffle(order)
    if order == second_order:
        second_order = second_order[::-1]

    first_build = _build(order)
    second_build = _build(second_order)

    assert first_build == second_build
    assert "not-shared" in first_build[3]
    assert "showing 8 of 21 characters" in first_build[4]
