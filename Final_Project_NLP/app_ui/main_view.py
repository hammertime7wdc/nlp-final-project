from __future__ import annotations

import asyncio

import flet as ft

from services.predictor import PredictionServiceError, predict_message, preload_model
from .result_panel import hide_result_panel
from .theme.palette import (
    ACCENT,
    ACCENT_SOFT,
    BG_CARD,
    BG_CARD_ALT,
    BG_MAIN,
    BORDER,
    TEXT_MAIN,
    TEXT_MUTED,
)


def main(page: ft.Page) -> None:
    page.title = "Spam Detection"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = BG_MAIN
    page.window_min_width = 800
    page.window_min_height = 700
    page.padding = 0
    page.scroll = ft.ScrollMode.AUTO

    selected_model = {"value": "BERT"}
    hovered_model = {"value": None}
    classifying_state = {"value": False}
    classify_request_id = {"value": 0}
    spam_threshold = 0.65

    title_icon = ft.Container(
        width=72,
        height=72,
        border_radius=20,
        bgcolor=ACCENT_SOFT,
        shadow=ft.BoxShadow(blur_radius=28, color="#553049", spread_radius=1),
        content=ft.Icon(ft.Icons.SHIELD_OUTLINED, color=TEXT_MAIN, size=34),
        alignment=ft.Alignment(0, 0),
    )

    heading = ft.Column(
        spacing=4,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        controls=[
            title_icon,
            ft.Text("Spam Detection", size=48, weight=ft.FontWeight.W_700, color=TEXT_MAIN),
            ft.Text(
                "Advanced transformer-based message classification",
                size=17,
                color=TEXT_MUTED,
            ),
        ],
    )

    card_title = ft.Row(
        spacing=8,
        controls=[
            ft.Icon(ft.Icons.HUB_OUTLINED, size=16, color=ACCENT),
            ft.Text("MODEL ARCHITECTURE", size=16, weight=ft.FontWeight.W_600, color=TEXT_MUTED),
        ],
    )

    model_cards: list[ft.Container] = []

    def make_model_card(name: str, subtitle: str, icon: str) -> ft.Container:
        card = ft.Container(
            data=name,
            expand=1,
            height=118,
            scale=1,
            border_radius=18,
            border=ft.Border.all(1, BORDER),
            bgcolor=BG_CARD_ALT,
            padding=ft.Padding.symmetric(horizontal=16, vertical=12),
            on_click=lambda e: on_model_change(e.control.data),
            on_hover=on_model_hover,
            animate=ft.Animation(220, ft.AnimationCurve.EASE_OUT),
            animate_scale=ft.Animation(180, ft.AnimationCurve.EASE_OUT),
            content=ft.Column(
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=6,
                controls=[
                    ft.Row(
                        spacing=8,
                        controls=[
                            ft.Icon(icon, color=ACCENT, size=20),
                            ft.Text(name, size=20, weight=ft.FontWeight.W_700, color=TEXT_MAIN),
                        ],
                    ),
                    ft.Text(subtitle, size=13, color=TEXT_MUTED, weight=ft.FontWeight.W_500),
                ],
            ),
        )
        model_cards.append(card)
        return card

    def on_model_hover(e: ft.ControlEvent) -> None:
        is_hovered = e.data == "true"
        hovered_model["value"] = e.control.data if is_hovered else None
        refresh_model_cards()
        page.update()

    def refresh_model_cards() -> None:
        for card in model_cards:
            is_selected = card.data == selected_model["value"]
            is_hovered = card.data == hovered_model["value"]

            if is_selected:
                card.bgcolor = ACCENT_SOFT
                card.border = ft.Border.all(1, ACCENT)
                card.shadow = ft.BoxShadow(blur_radius=20, spread_radius=0, color="#5f2b4b")
                card.scale = 1.015
            elif is_hovered:
                card.bgcolor = "#3a3d49"
                card.border = ft.Border.all(1, "#7b4a67")
                card.shadow = ft.BoxShadow(blur_radius=12, spread_radius=0, color="#3e3140")
                card.scale = 1.02
            else:
                card.bgcolor = BG_CARD_ALT
                card.border = ft.Border.all(1, BORDER)
                card.shadow = None
                card.scale = 1

    def on_model_change(name: str) -> None:
        selected_model["value"] = name
        selected_model_label.value = name
        clear_error()

        # Clear previous classification result when the model changes.
        page.run_task(hide_result_panel, result_panel, page)

        refresh_model_cards()
        page.update()

        async def preload_task(_: ft.ControlEvent) -> None:
            try:
                await asyncio.to_thread(preload_model, name)
            except PredictionServiceError as ex:
                show_error(str(ex))
                page.update()
            except Exception as ex:
                show_error(f"Failed to preload model: {ex}")
                page.update()

        # Pre-load the model when the user selects it
        page.run_task(preload_task, None)

    model_row = ft.ResponsiveRow(
        run_spacing=12,
        spacing=12,
        controls=[
            ft.Container(col={"sm": 12, "md": 4}, content=make_model_card("BERT", "110M - Standard", ft.Icons.PSYCHOLOGY_ALT_OUTLINED)),
            ft.Container(col={"sm": 12, "md": 4}, content=make_model_card("RoBERTa", "125M - Enhanced", ft.Icons.BOLT_OUTLINED)),
            ft.Container(col={"sm": 12, "md": 4}, content=make_model_card("DistilBERT", "66M - Fast", ft.Icons.SPEED_OUTLINED)),
        ],
    )

    input_title = ft.Row(
        spacing=8,
        controls=[
            ft.Icon(ft.Icons.MAIL_OUTLINED, size=16, color=ACCENT),
            ft.Text("MESSAGE CONTENT", size=16, weight=ft.FontWeight.W_600, color=TEXT_MUTED),
        ],
    )

    message_input = ft.TextField(
        hint_text="Paste or type the message you want to classify...",
        multiline=True,
        min_lines=6,
        max_lines=10,
        expand=True,
        border_radius=16,
        border_color=BORDER,
        focused_border_color=ACCENT,
        bgcolor="#474a55",
        color=TEXT_MAIN,
        hint_style=ft.TextStyle(color="#8d92a3", size=15),
        text_size=15,
        content_padding=ft.Padding.symmetric(horizontal=12, vertical=12),
    )

    char_counter = ft.Text("0 characters", color=TEXT_MUTED, size=12)

    submit_hint = ft.Container(
        padding=ft.Padding.symmetric(horizontal=8, vertical=4),
        border=ft.Border.all(1, BORDER),
        border_radius=10,
        content=ft.Text("Ctrl + Enter", color=TEXT_MUTED, size=11),
    )

    classify_btn = ft.Button(
        "Classify Message",
        icon=ft.Icons.AUTO_AWESOME,
        disabled=True,
        style=ft.ButtonStyle(
            bgcolor={ft.ControlState.DEFAULT: "#40434f", ft.ControlState.DISABLED: "#3b3d47"},
            color={ft.ControlState.DEFAULT: "#d7dae6", ft.ControlState.DISABLED: "#8f93a1"},
            shape=ft.RoundedRectangleBorder(radius=10),
            text_style=ft.TextStyle(size=16, weight=ft.FontWeight.W_600),
            padding=ft.Padding.symmetric(horizontal=40, vertical=12),
        ),
    )

    classify_btn_wrap = ft.Container(
        content=classify_btn,
        scale=1,
        animate_scale=ft.Animation(180, ft.AnimationCurve.EASE_OUT),
    )

    error_banner = ft.Container(
        visible=False,
        padding=ft.Padding.symmetric(horizontal=12, vertical=10),
        border_radius=10,
        border=ft.Border.all(1, "#7a3a4e"),
        bgcolor="#432634",
        content=ft.Row(
            spacing=8,
            controls=[
                ft.Icon(ft.Icons.ERROR_OUTLINE, size=18, color="#ff9bb4"),
                ft.Text("", color="#ffd9e5", size=13),
            ],
        ),
    )

    result_icon = ft.Container(
        width=78,
        height=78,
        border_radius=18,
        bgcolor=ACCENT_SOFT,
        alignment=ft.Alignment(0, 0),
        shadow=ft.BoxShadow(blur_radius=16, color="#5f2b4b"),
        content=ft.Icon(ft.Icons.CHECK, size=38, color=TEXT_MAIN),
    )

    result_label = ft.Text("Ham", size=42, weight=ft.FontWeight.W_700, color=TEXT_MAIN)
    result_subtitle = ft.Text("Legitimate message verified", size=16, color=TEXT_MUTED)
    confidence_value = ft.Text("74.7%", size=52, weight=ft.FontWeight.W_300, color=TEXT_MAIN)
    confidence_bar = ft.ProgressBar(value=0.747, color=ACCENT, bgcolor="#6d6672", bar_height=20)
    ham_probability_text = ft.Text("Ham: 50.0%", size=13, color=TEXT_MUTED)
    spam_probability_text = ft.Text("Spam: 50.0%", size=13, color=TEXT_MUTED)
    threshold_text = ft.Text(f"Spam threshold: {spam_threshold:.0%}", size=12, color=TEXT_MUTED)

    selected_model_label = ft.Text("BERT", size=18, weight=ft.FontWeight.W_600, color=TEXT_MAIN)
    proc_time_label = ft.Text("877ms", size=18, weight=ft.FontWeight.W_600, color=TEXT_MAIN)
    status_label = ft.Row(
        spacing=6,
        controls=[
            ft.Container(width=8, height=8, border_radius=4, bgcolor=ACCENT),
            ft.Text("Complete", size=18, weight=ft.FontWeight.W_600, color=TEXT_MAIN),
        ],
    )

    result_panel = ft.Container(
        visible=False,
        opacity=0,
        offset=ft.Offset(0, 0.06),
        scale=0.97,
        animate_opacity=ft.Animation(340, ft.AnimationCurve.EASE_OUT),
        animate_offset=ft.Animation(340, ft.AnimationCurve.EASE_OUT),
        animate_scale=ft.Animation(340, ft.AnimationCurve.EASE_OUT),
        margin=ft.Margin.only(top=16),
        padding=ft.Padding.all(20),
        border_radius=24,
        border=ft.Border.all(1, BORDER),
        bgcolor=BG_CARD,
        content=ft.Column(
            spacing=12,
            controls=[
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                    controls=[
                        ft.Row(
                            spacing=12,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            controls=[
                                result_icon,
                                ft.Column(
                                    spacing=4,
                                    controls=[
                                        ft.Text("CLASSIFICATION RESULT", size=12, color=TEXT_MUTED),
                                        result_label,
                                        result_subtitle,
                                    ],
                                ),
                            ],
                        ),
                        ft.Column(
                            horizontal_alignment=ft.CrossAxisAlignment.END,
                            spacing=4,
                            controls=[
                                ft.Text("CONFIDENCE", size=12, color=TEXT_MUTED),
                                confidence_value,
                            ],
                        ),
                    ],
                ),
                confidence_bar,
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    controls=[ham_probability_text, spam_probability_text, threshold_text],
                ),
                ft.Divider(color="#4d515d"),
                ft.ResponsiveRow(
                    controls=[
                        ft.Container(
                            col={"sm": 12, "md": 4},
                            content=ft.Column(
                                spacing=5,
                                controls=[
                                    ft.Text("MODEL", size=12, color=TEXT_MUTED),
                                    selected_model_label,
                                ],
                            ),
                        ),
                        ft.Container(
                            col={"sm": 12, "md": 4},
                            content=ft.Column(
                                spacing=5,
                                controls=[
                                    ft.Text("PROCESSING TIME", size=12, color=TEXT_MUTED),
                                    proc_time_label,
                                ],
                            ),
                        ),
                        ft.Container(
                            col={"sm": 12, "md": 4},
                            content=ft.Column(
                                spacing=5,
                                controls=[
                                    ft.Text("STATUS", size=12, color=TEXT_MUTED),
                                    status_label,
                                ],
                            ),
                        ),
                    ]
                ),
            ],
        ),
    )

    footer_chip = ft.Container(
        margin=ft.Margin.only(top=14, bottom=14),
        padding=ft.Padding.symmetric(horizontal=12, vertical=6),
        border=ft.Border.all(1, BORDER),
        border_radius=16,
        content=ft.Row(
            tight=True,
            spacing=8,
            controls=[
                ft.Container(width=6, height=6, border_radius=3, bgcolor=ACCENT),
                ft.Text("Powered by transformer-based contextual embeddings", size=12, color=TEXT_MUTED),
            ],
        ),
    )

    def show_error(message: str) -> None:
        error_banner.visible = True
        error_banner.content.controls[1].value = message

    def clear_error() -> None:
        error_banner.visible = False
        error_banner.content.controls[1].value = ""

    def update_input_state() -> None:
        text = message_input.value or ""
        char_counter.value = f"{len(text)} characters"
        classify_btn.disabled = len(text.strip()) == 0 or classifying_state["value"]

    def on_classify_btn_hover(e: ft.ControlEvent) -> None:
        if classify_btn.disabled or classifying_state["value"]:
            classify_btn_wrap.scale = 1
        else:
            classify_btn_wrap.scale = 1.06 if e.data == "true" else 1
        page.update()

    def set_classify_loading(is_loading: bool) -> None:
        classifying_state["value"] = is_loading
        classify_btn.text = "Classifying..." if is_loading else "Classify Message"
        classify_btn.icon = ft.Icons.HOURGLASS_TOP if is_loading else ft.Icons.AUTO_AWESOME
        classify_btn_wrap.scale = 1
        update_input_state()

    async def render_result(
        label: str,
        confidence: float,
        ham_probability: float,
        spam_probability: float,
        processing_time_ms: int,
        model_name: str,
    ) -> None:
        is_spam = spam_probability >= spam_threshold
        is_ham = not is_spam
        result_label.value = "Ham" if is_ham else "Spam"
        result_subtitle.value = (
            "Legitimate message verified" if is_ham else "Potential spam content detected"
        )
        result_icon.bgcolor = "#7c3a62" if is_ham else ACCENT_SOFT
        result_icon.content = ft.Icon(
            ft.Icons.CHECK if is_ham else ft.Icons.WARNING_AMBER_ROUNDED,
            size=56,
            color=TEXT_MAIN,
        )

        confidence_pct = max(0.0, min(1.0, confidence))
        confidence_value.value = f"{confidence_pct * 100:.1f}%"
        confidence_bar.value = confidence_pct
        confidence_bar.color = "#c34f86" if is_ham else ACCENT
        selected_model_label.value = model_name
        proc_time_label.value = f"{processing_time_ms}ms"
        ham_probability_text.value = f"Ham: {ham_probability * 100:.1f}%"
        spam_probability_text.value = f"Spam: {spam_probability * 100:.1f}%"
        threshold_text.value = f"Spam threshold: {spam_threshold:.0%}"
        if not result_panel.visible:
            result_panel.visible = True
            result_panel.opacity = 0
            result_panel.offset = ft.Offset(0, 0.06)
            result_panel.scale = 0.97
            page.update()
            await asyncio.sleep(0.03)

        result_panel.opacity = 1
        result_panel.offset = ft.Offset(0, 0)
        result_panel.scale = 1
        page.update()

    def on_input_change(_: ft.ControlEvent) -> None:
        update_input_state()
        page.update()

    async def on_classify(_: ft.ControlEvent) -> None:
        text = (message_input.value or "").strip()
        if not text or classifying_state["value"]:
            return

        clear_error()
        classify_request_id["value"] += 1
        request_id = classify_request_id["value"]

        set_classify_loading(True)
        page.update()

        # Small delay so users can perceive loading feedback before result appears.
        await asyncio.sleep(0.35)

        try:
            model_name = selected_model["value"]
            prediction = await asyncio.to_thread(predict_message, text, model_name)

            # Ignore stale results if a newer classify request was started.
            if request_id != classify_request_id["value"]:
                return

            await render_result(
                label=prediction.label,
                confidence=prediction.confidence,
                ham_probability=prediction.ham_probability,
                spam_probability=prediction.spam_probability,
                processing_time_ms=prediction.processing_time_ms,
                model_name=model_name,
            )
        except PredictionServiceError as ex:
            show_error(str(ex))
        except Exception as ex:
            show_error(f"Classification failed: {ex}")
        finally:
            if request_id == classify_request_id["value"]:
                set_classify_loading(False)
            page.update()

    message_input.on_change = on_input_change
    classify_btn.on_click = lambda _: page.run_task(on_classify, None)
    classify_btn_wrap.on_hover = on_classify_btn_hover

    def on_keyboard(e: ft.KeyboardEvent) -> None:
        if (e.ctrl or e.meta) and e.key == "Enter":
            page.run_task(on_classify, None)

    page.on_keyboard_event = on_keyboard

    refresh_model_cards()
    update_input_state()

    content_block = ft.Container(
        width=1120,
        margin=ft.Margin.symmetric(horizontal=20, vertical=12),
        padding=ft.Padding.all(16),
        border=ft.Border.all(1, BORDER),
        border_radius=24,
        bgcolor=BG_CARD,
        content=ft.Column(
            spacing=12,
            controls=[
                card_title,
                model_row,
                input_title,
                message_input,
                error_banner,
                ft.Row(
                    expand=True,
                    alignment=ft.MainAxisAlignment.END,
                    spacing=10,
                    controls=[char_counter, submit_hint],
                ),
                ft.Row(
                    alignment=ft.MainAxisAlignment.CENTER,
                    controls=[classify_btn_wrap],
                ),
            ],
        ),
    )

    gradient_background = ft.Container(
        expand=True,
        padding=0,
        gradient=ft.LinearGradient(
            begin=ft.Alignment(-1, -1),
            end=ft.Alignment(1, 1),
            colors=["#1a1b22", "#20232c", "#1a1b22"],
        ),
        content=ft.Column(
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=0,
            controls=[
                ft.Container(height=10),
                ft.Container(width=1120, alignment=ft.Alignment(0, 0), content=heading),
                content_block,
                ft.Container(width=1120, margin=ft.Margin.symmetric(horizontal=20), content=result_panel),
                ft.Container(width=1120, alignment=ft.Alignment(0, 0), content=footer_chip),
            ],
        ),
    )

    page.add(gradient_background)
