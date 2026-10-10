"""Integration checks exercise actual Streamlit widgets against isolated databases."""

from bnivoice360.config import Settings
from bnivoice360.container import create_services


def navigate(app, title):
    app.sidebar.radio(key="current_page").set_value(title).run()
    assert not app.exception


def test_every_page_renders_and_roles_hide_admin_navigation(streamlit_app):
    app = streamlit_app
    assert app.metric[0].value == "96"
    assert "Data Ingestion" not in app.sidebar.radio(key="current_page").options
    for title in (
        "Issue Analytics & Voice Feed",
        "Graph Analytics",
        "BNIVoice360 Companion",
        "Home",
    ):
        navigate(app, title)
    app.sidebar.selectbox(key="selected_role").set_value("Data Admin").run()
    assert not app.exception
    navigate(app, "Data Ingestion")
    assert len(app.get("file_uploader")) == 8
    app.sidebar.selectbox(key="selected_role").set_value("Manager / Executive").run()
    assert not app.exception
    assert app.sidebar.radio(key="current_page").value == "Home"
    assert "Data Ingestion" not in app.sidebar.radio(key="current_page").options


def test_crawl_save_predict_updates_data_and_clears_saved_preview(streamlit_app, tmp_path):
    app = streamlit_app
    app.sidebar.selectbox(key="selected_role").set_value("Data Admin").run()
    navigate(app, "Data Ingestion")
    app.button(key="crawl_X").click().run()
    assert not app.exception
    assert len(app.session_state["crawl_data_X"]) == 10
    app.button(key="save_crawl_X").click().run()
    assert not app.exception
    assert "crawl_data_X" not in app.session_state
    services = create_services(Settings(tmp_path / "ui.db"))
    assert len(services.repository.load()) == 106
    app.button(key="predict_X").click().run()
    assert not app.exception
    data = services.repository.load()
    assert data[data["source"] == "X"]["sentiment"].notna().all()
    # Data health metrics update on the same rerun as the write.
    assert app.metric[2].value == "0"


def test_escalation_button_persists_state(streamlit_app, tmp_path):
    app = streamlit_app
    navigate(app, "Issue Analytics & Voice Feed")
    action = next(button for button in app.button if button.label == "Escalate")
    record_id = action.key.removeprefix("esc_")
    action.click().run()
    assert not app.exception
    services = create_services(Settings(tmp_path / "ui.db"))
    data = services.repository.load().set_index("suara_nasabah_id")
    assert data.loc[record_id, "is_escalated"] == 1


def test_companion_chat_history_survives_navigation(streamlit_app):
    app = streamlit_app
    navigate(app, "BNIVoice360 Companion")
    app.chat_input[0].set_value("Berapa urgent IT issue yang belum dieskalasi?").run()
    assert not app.exception
    messages = list(app.session_state["chat_history"])
    assert len(messages) == 3
    assert messages[-1]["role"] == "assistant"
    assert "voice" in messages[-1]["content"]
    navigate(app, "Home")
    navigate(app, "BNIVoice360 Companion")
    assert app.session_state["chat_history"] == messages
