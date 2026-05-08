"""static_render.py — Bridge between Tab.render() and static HTML serialization.

Converts the list[go.Figure] returned by ``tab.render(ctx, container=None)``
into an HTML fragment suitable for embedding in the Jinja2 report template.

No Streamlit dependency — this module must be importable in a headless context.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from lerobot_isaac_dashboard.tabs._base import Tab, TabContext


def render_tab_to_html(
    tab: "Tab",
    ctx: "TabContext",
    *,
    include_plotlyjs: str | bool = False,
) -> tuple[str, list[str]]:
    """Run ``tab.render(ctx, container=None)`` and serialize figures to HTML.

    Parameters
    ----------
    tab:
        Instantiated Tab object (not the class).
    ctx:
        Shared ``TabContext`` built by ``run_loaders_headless`` or the live app.
    include_plotlyjs:
        Passed directly to ``plotly.io.to_html`` for the *first* figure only.
        All subsequent figures use ``include_plotlyjs=False``.

        Use ``False`` (default) when the caller (report.py) injects Plotly JS
        once at the document level.  Use ``"inline"`` or ``"cdn"`` to make
        a standalone fragment.

    Returns
    -------
    (html_body, warnings)
        ``html_body`` is a concatenation of per-figure HTML fragments.
        ``warnings`` is a list of non-fatal issue strings (empty on success).
    """
    warnings: list[str] = []

    # Collect figures from the tab
    try:
        figs = tab.render(ctx, container=None)
    except Exception as exc:  # noqa: BLE001
        msg = f"Tab '{tab.title}' raised during static render: {exc}"
        logger.warning(msg)
        warnings.append(msg)
        return "", warnings

    if not figs:
        return "", warnings

    # Serialize each figure to an HTML fragment
    try:
        import plotly.io as pio
    except ImportError:
        msg = "plotly is not installed; cannot serialize figures to HTML"
        logger.error(msg)
        warnings.append(msg)
        return "", warnings

    html_parts: list[str] = []
    for i, fig in enumerate(figs):
        try:
            # Only the first figure carries the plotlyjs flag; the rest never
            # re-include it.  The caller controls whether JS goes here or in
            # the <head>.
            fig_include_js = include_plotlyjs if i == 0 else False
            html_part = pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs=fig_include_js,
                config={"responsive": True},
            )
            html_parts.append(html_part)
        except Exception as exc:  # noqa: BLE001
            msg = f"Tab '{tab.title}' fig[{i}] serialization failed: {exc}"
            logger.warning(msg)
            warnings.append(msg)

    return "\n".join(html_parts), warnings
