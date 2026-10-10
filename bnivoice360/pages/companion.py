"""Chat presentation; querying and answers belong to CompanionService."""

import streamlit as st

from bnivoice360.container import AppServices
from bnivoice360.ui.components import header


def render(services: AppServices) -> None:
    header(
        "BNIVoice360 Companion",
        "Tanyakan apa pun tentang sentimen, emerging issues, dan status tindak lanjut.",
        "Executive AI Companion",
    )
    with st.expander("Contoh pertanyaan"):
        st.markdown(
            "- Apa top 3 topic negatif dari PlayStore minggu ini?\n"
            "- Berapa urgent IT issue yang belum dieskalasi?\n"
            "- Ringkas sentimen nasabah mengenai Mobile Banking dan QRIS."
        )
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Tanyakan insight dari database VoC…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        answer = services.companion.answer(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
