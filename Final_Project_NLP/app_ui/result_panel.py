from __future__ import annotations

import asyncio

import flet as ft


async def hide_result_panel(result_panel: ft.Container, page: ft.Page) -> None:
    """Fade out and collapse the current result panel."""
    if not result_panel.visible:
        return

    result_panel.opacity = 0
    result_panel.offset = ft.Offset(0, 0.03)
    result_panel.scale = 0.98
    page.update()

    # Let the fade animation finish before collapsing layout space.
    await asyncio.sleep(0.18)

    result_panel.visible = False
    page.update()
