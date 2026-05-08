"""_compare.py — CompareWrapper: wraps a Tab class in a side-by-side renderer.

This module is a Streamlit-side helper that adapts the existing Tab interface
for the Compare mode sidebar added in Phase 8.  It is NOT used in the static
export path (compare.py handles that directly).

Usage inside app.py::

    from lerobot_isaac_dashboard.tabs._compare import CompareWrapper

    # 2-way
    CompareWrapper(tab_cls).render_2way(ctx_a, ctx_b, container=st.container())

    # N-way
    CompareWrapper(tab_cls).render_nway([ctx_a, ctx_b, ctx_c], labels, container=st.container())
"""

from __future__ import annotations

from typing import Any


class CompareWrapper:
    """Wraps a Tab class to render it in side-by-side or overlay mode.

    Parameters
    ----------
    tab_cls:
        The Tab subclass (not an instance) to wrap.
    """

    def __init__(self, tab_cls: Any) -> None:
        self.tab_cls = tab_cls

    # ------------------------------------------------------------------
    # 2-way side-by-side
    # ------------------------------------------------------------------

    def render_2way(
        self,
        ctx_a: Any,
        ctx_b: Any,
        *,
        label_a: str = "A",
        label_b: str = "B",
        container: Any = None,
    ) -> tuple[list[Any], list[Any]]:
        """Render this tab for both contexts in two columns.

        Parameters
        ----------
        ctx_a:
            TabContext for snapshot A.
        ctx_b:
            TabContext for snapshot B.
        label_a:
            Display label for column A.
        label_b:
            Display label for column B.
        container:
            Streamlit container to render into.  When None, returns figs only.

        Returns
        -------
        (figs_a, figs_b)
            Lists of figures from each side.
        """
        try:
            import streamlit as st

            _has_st = True
        except ImportError:
            _has_st = False

        tab_a = self.tab_cls()
        tab_b = self.tab_cls()

        if container is not None and _has_st:
            import streamlit as st

            col_a, col_b = container.columns(2)
            col_a.markdown(f"**{label_a}**")
            col_b.markdown(f"**{label_b}**")

            figs_a = tab_a.render(ctx_a, container=None)
            figs_b = tab_b.render(ctx_b, container=None)

            for fig in figs_a:
                try:
                    col_a.plotly_chart(fig, use_container_width=True)
                except Exception:  # noqa: BLE001
                    pass
            for fig in figs_b:
                try:
                    col_b.plotly_chart(fig, use_container_width=True)
                except Exception:  # noqa: BLE001
                    pass
        else:
            figs_a = tab_a.render(ctx_a, container=None)
            figs_b = tab_b.render(ctx_b, container=None)

        return figs_a, figs_b

    # ------------------------------------------------------------------
    # N-way overlay
    # ------------------------------------------------------------------

    def render_nway(
        self,
        contexts: list[Any],
        labels: list[str],
        *,
        container: Any = None,
    ) -> list[Any]:
        """Overlay this tab's figures from multiple contexts.

        Parameters
        ----------
        contexts:
            List of TabContext objects, one per snapshot.
        labels:
            Display labels (same order as contexts).
        container:
            Streamlit container.  When None, returns figs only.

        Returns
        -------
        list[go.Figure]
            Overlay figures with one trace per snapshot.
        """
        try:
            import plotly.graph_objects as go

            _has_plotly = True
        except ImportError:
            _has_plotly = False

        try:
            import streamlit as st

            _has_st = True
        except ImportError:
            _has_st = False

        if not _has_plotly:
            return []

        # Collect per-snapshot figure lists
        per_snap: list[tuple[str, list[Any]]] = []
        for ctx, label in zip(contexts, labels):
            try:
                figs = self.tab_cls().render(ctx, container=None)
            except Exception:  # noqa: BLE001
                figs = []
            per_snap.append((label, figs))

        max_figs = max((len(figs) for _, figs in per_snap), default=0)
        overlay_figs: list[Any] = []

        for fig_idx in range(max_figs):
            overlay = go.Figure()
            has_any = False
            for snap_label, figs in per_snap:
                if fig_idx >= len(figs):
                    continue
                src_fig = figs[fig_idx]
                for trace in src_fig.data:
                    new_trace = trace.__class__(
                        **{k: v for k, v in trace.to_plotly_json().items()
                           if k not in ("type",)},
                        name=f"{snap_label} – {trace.name or ''}".strip(" –"),
                        showlegend=True,
                    )
                    overlay.add_trace(new_trace)
                    has_any = True
            if has_any:
                # Copy layout from first available figure
                for _, figs in per_snap:
                    if fig_idx < len(figs):
                        try:
                            overlay.update_layout(figs[fig_idx].layout)
                        except Exception:  # noqa: BLE001
                            pass
                        break
                overlay.update_layout(showlegend=True)
                overlay_figs.append(overlay)

        if container is not None and _has_st:
            import streamlit as st

            for fig in overlay_figs:
                try:
                    container.plotly_chart(fig, use_container_width=True)
                except Exception:  # noqa: BLE001
                    pass

        return overlay_figs
