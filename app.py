"""
Gmail AI Pro Suite
==================
A professional email intelligence suite for Gmail, built with Streamlit.

Key capabilities:
  - Direct, live IMAP/SMTP connection to Gmail (zero-database architecture)
  - AI-generated Executive Digest across the full inbox
  - Line-by-line, single-sentence highlights for every message
  - Automated spam and promotional-email quarantine
  - One-click AI reply drafting and sending

No email content, credentials, or summaries are ever written to disk or an
external database; everything lives in the Streamlit session for the
duration of the browser session only.
"""

import io
import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
import pandas as pd

from gmail_service import (
    GmailClient,
    get_mock_emails,
    extract_pdf_text_from_bytes,
    clean_html_to_text,
)
from ai_summarizer import (
    analyze_single_email,
    generate_inbox_master_digest,
    generate_ai_reply,
    classify_topic,
    detect_spam_and_promotions,
)

# Plotly is optional — the app degrades gracefully to a bar chart if it isn't installed.
try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Gmail AI Pro Suite",
    page_icon="📬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown("""
<style>
    :root {
        --accent: #6366f1;
        --accent-dark: #818cf8;
    }

    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Metric cards and bordered containers pick up the app's own theme colors,
       so they match the sidebar's boxed look in both light and dark mode. */
    div[data-testid="stMetric"],
    div[data-testid="metric-container"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-left: 3px solid var(--accent);
        padding: 12px 18px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--secondary-background-color);
        border-radius: 12px;
    }

    .badge {
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-urgent   { background-color: rgba(220, 38, 38, 0.18);  color: #fca5a5; }
    .badge-category { background-color: rgba(99, 102, 241, 0.18); color: #a5b4fc; font-weight: 500; }
    .badge-clean    { background-color: rgba(16, 185, 129, 0.18); color: #6ee7b7; }
    .badge-spam     { background-color: rgba(225, 29, 72, 0.20);  color: #fda4af; font-weight: 700; }
    .badge-promo    { background-color: rgba(217, 119, 6, 0.20);  color: #fcd34d; }
    .badge-unread   { background-color: rgba(124, 58, 237, 0.20); color: #c4b5fd; }

    .line-card, .line-card-spam, .line-card-urgent {
        background-color: var(--secondary-background-color);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.18);
    }
    .line-card        { border-left: 5px solid var(--accent); }
    .line-card-spam   { border-left: 5px solid #e11d48; }
    .line-card-urgent { border-left: 5px solid #dc2626; }

    /* Light mode gets slightly darker text on the lighter pastel badges above still
       read well; fall back to solid pastels when the app is explicitly light-themed. */
    @media (prefers-color-scheme: light) {
        .badge-urgent   { background-color: #fef2f2; color: #b91c1c; }
        .badge-category { background-color: #eef2ff; color: var(--accent); }
        .badge-clean    { background-color: #ecfdf5; color: #047857; }
        .badge-spam     { background-color: #fff1f2; color: #be123c; }
        .badge-promo    { background-color: #fffbeb; color: #b45309; }
        .badge-unread   { background-color: #f5f3ff; color: #6d28d9; }
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Session state (in-memory only — nothing is persisted between sessions)
# --------------------------------------------------------------------------
DEFAULT_STATE = {
    "emails": None,               # populated below via get_mock_emails()
    "gmail_connected": False,
    "gmail_user": "",
    "gmail_pass": "",
    "selected_email_id": "mock_1",
    "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
    "reply_draft": "",
    "current_folder": "INBOX",
    "fetch_limit": 30,
}
for key, default in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = get_mock_emails() if key == "emails" else default


# --------------------------------------------------------------------------
# Sidebar — connection management and navigation
# --------------------------------------------------------------------------
st.sidebar.markdown("## 📬 Gmail AI Pro Suite")
st.sidebar.caption("Live Gmail via IMAP/SMTP · AI Digest · Spam Isolation · Line Highlights")

if st.session_state["gmail_connected"]:
    st.sidebar.success(f"Connected: `{st.session_state['gmail_user']}`")
    if st.sidebar.button("Disconnect", use_container_width=True):
        st.session_state["gmail_connected"] = False
        st.session_state["gmail_user"] = ""
        st.session_state["gmail_pass"] = ""
        st.session_state["emails"] = get_mock_emails()
        st.rerun()
else:
    st.sidebar.info("Demo mode active — connect a live Gmail account below.")

with st.sidebar.expander("Connect a Gmail account", expanded=not st.session_state["gmail_connected"]):
    st.markdown("Enter your Gmail address and a 16-character **Google App Password**.")
    input_email = st.text_input(
        "Gmail address", value=st.session_state.get("gmail_user", ""), placeholder="you@gmail.com"
    )
    input_pass = st.text_input(
        "Google App Password",
        value=st.session_state.get("gmail_pass", ""),
        type="password",
        placeholder="xxxx xxxx xxxx xxxx",
    )

    col_folder1, col_folder2 = st.columns(2)
    with col_folder1:
        folder_choice = st.selectbox(
            "Folder", ["INBOX", "[Gmail]/Spam", "[Gmail]/Starred", "[Gmail]/Sent Mail"], index=0
        )
    with col_folder2:
        fetch_limit = st.selectbox("Fetch limit", [15, 30, 50, 100], index=1)
        st.session_state["fetch_limit"] = fetch_limit

    col_conn1, col_conn2 = st.columns(2)
    with col_conn1:
        if st.button("Connect", use_container_width=True, type="primary"):
            if not input_email or not input_pass:
                st.error("Please provide both an email address and an app password.")
            else:
                with st.spinner("Verifying Gmail connection..."):
                    client = GmailClient(input_email, input_pass)
                    ok, msg = client.test_connection()
                    if ok:
                        st.session_state["gmail_connected"] = True
                        st.session_state["gmail_user"] = input_email.strip()
                        st.session_state["gmail_pass"] = input_pass.strip()
                        st.session_state["current_folder"] = folder_choice
                        with st.spinner(f"Fetching messages from {folder_choice}..."):
                            fetched = client.fetch_emails(folder=folder_choice, limit=fetch_limit)
                            st.session_state["emails"] = fetched or []
                            if fetched:
                                st.session_state["selected_email_id"] = fetched[0]["id"]
                        st.success("Connected to Gmail successfully.")
                        st.rerun()
                    else:
                        st.error(msg)
    with col_conn2:
        if st.button("Sync now", use_container_width=True, disabled=not st.session_state["gmail_connected"]):
            with st.spinner(f"Syncing {folder_choice}..."):
                client = GmailClient(st.session_state["gmail_user"], st.session_state["gmail_pass"])
                fetched = client.fetch_emails(folder=folder_choice, limit=st.session_state.get("fetch_limit", 30))
                if fetched:
                    st.session_state["emails"] = fetched
                    st.session_state["selected_email_id"] = fetched[0]["id"]
            st.success("Sync complete.")
            st.rerun()

    with st.expander("How to generate a Google App Password"):
        st.markdown("""
        1. Open [Google Account Security](https://myaccount.google.com/security).
        2. Turn on **2-Step Verification** if it isn't already enabled.
        3. Search for **"App passwords"** in the top search bar.
        4. Name it `Gmail AI` and select **Create**.
        5. Copy the generated 16-character code and paste it above.
        """)

st.sidebar.divider()

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "AI Master Digest",
        "Line-by-Line Highlights",
        "Spam & Promotions",
        "Inbox & Detail View",
        "Urgent & Action Required",
        "AI Reply Composer",
        "Import Files (JSON/CSV/PDF)",
        "Settings",
    ],
    index=0,
)

all_emails = st.session_state.get("emails", [])
total_cnt = len(all_emails)
unread_cnt = sum(1 for m in all_emails if m.get("unread", False))

st.sidebar.divider()
st.sidebar.markdown(f"**Loaded:** `{total_cnt}` &nbsp;·&nbsp; **Unread:** `{unread_cnt}`")


# ==========================================================================
# VIEW: AI MASTER DIGEST
# ==========================================================================
if nav_choice == "AI Master Digest":
    st.markdown("## AI Master Inbox Digest")
    st.caption("A full-inbox executive summary with priority actions, category breakdown, and key dates.")

    if not all_emails:
        st.warning("No emails available to summarize. Connect a Gmail account or import a file to get started.")
    else:
        with st.spinner("Generating executive digest..."):
            digest = generate_inbox_master_digest(all_emails, gemini_api_key=st.session_state.get("gemini_api_key"))

        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Total analyzed", digest["total_emails"])
        col_m2.metric("Clean / important", len(digest["clean_emails"]))
        col_m3.metric("Urgent actions", len(digest["urgent_emails"]))
        col_m4.metric("Quarantined spam", len(digest["spam_emails"]))
        col_m5.metric("Promotions", len(digest["promo_emails"]))

        st.divider()

        with st.container(border=True):
            st.markdown("### Executive Summary")
            st.markdown(digest["executive_summary"])

        col_left, col_right = st.columns([1.3, 1])

        with col_left:
            st.markdown("### Priority Actions Required")
            if digest["urgent_emails"]:
                for u in digest["urgent_emails"]:
                    with st.container(border=True):
                        c_u1, c_u2 = st.columns([3, 1])
                        with c_u1:
                            st.markdown(f"**{u['subject']}**")
                            st.caption(f"From: `{u['sender']}` · {u['date']}")
                            st.markdown(f"**Action:** {u['action']}")
                        with c_u2:
                            st.markdown(f"<span class='badge badge-urgent'>{u['urgency']}</span>", unsafe_allow_html=True)
                            st.markdown(f"<span class='badge badge-category'>{u['topic']}</span>", unsafe_allow_html=True)
            else:
                st.success("No urgent actions pending.")

            st.markdown("### Action Items")
            if digest["action_items"]:
                df_actions = pd.DataFrame(digest["action_items"])
                st.dataframe(
                    df_actions[["task", "from", "topic"]].rename(
                        columns={"task": "Action Item", "from": "Requested By", "topic": "Category"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No direct action requests detected in message bodies.")

        with col_right:
            st.markdown("### Category Breakdown")
            if digest["category_counts"]:
                df_cat = pd.DataFrame(list(digest["category_counts"].items()), columns=["Category", "Count"])
                if HAS_PLOTLY:
                    fig = px.pie(
                        df_cat, names="Category", values="Count", hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Safe,
                    )
                    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(df_cat.set_index("Category"))

            st.markdown("### Key Dates & Deadlines")
            if digest["upcoming_dates"]:
                for d in digest["upcoming_dates"]:
                    st.markdown(f"- **`{d['date_mention']}`** — *{d['subject']}* (`{d['from']}`)")
            else:
                st.caption("No upcoming dates were extracted.")

        st.divider()

        st.markdown("### Export")
        st.download_button(
            label="Download Executive Digest (.md)",
            data=digest["markdown_report"],
            file_name=f"gmail_executive_digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            type="primary",
        )


# ==========================================================================
# VIEW: LINE-BY-LINE HIGHLIGHTS
# ==========================================================================
elif nav_choice == "Line-by-Line Highlights":
    st.markdown("## Line-by-Line Message Highlights")
    st.caption("A single-sentence takeaway, urgency level, and category for every message in the inbox.")

    if not all_emails:
        st.warning("No emails loaded.")
    else:
        with st.spinner("Extracting highlights..."):
            digest = generate_inbox_master_digest(all_emails)
            highlights = digest["line_by_line_highlights"]

        col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
        with col_f1:
            filter_cat = st.selectbox("Category", ["All Categories"] + sorted(list(digest["category_counts"].keys())))
        with col_f2:
            filter_type = st.selectbox(
                "Type",
                ["All Messages", "Clean & Work Only", "Urgent Only", "Spam Only", "Promotions Only"],
            )
        with col_f3:
            search_query = st.text_input("Search", placeholder="Search by keyword, sender, or subject...")

        filtered_hl = highlights
        if filter_cat != "All Categories":
            filtered_hl = [h for h in filtered_hl if h["category"] == filter_cat]
        if filter_type == "Clean & Work Only":
            filtered_hl = [h for h in filtered_hl if not h["is_spam"] and not h["is_promo"]]
        elif filter_type == "Urgent Only":
            filtered_hl = [h for h in filtered_hl if "High" in h["urgency"] or "Medium" in h["urgency"]]
        elif filter_type == "Spam Only":
            filtered_hl = [h for h in filtered_hl if h["is_spam"]]
        elif filter_type == "Promotions Only":
            filtered_hl = [h for h in filtered_hl if h["is_promo"]]

        if search_query:
            sq = search_query.lower()
            filtered_hl = [
                h for h in filtered_hl
                if sq in h["subject"].lower() or sq in h["sender"].lower() or sq in h["one_line_important"].lower()
            ]

        st.markdown(f"Showing **{len(filtered_hl)}** result(s).")

        for idx, h in enumerate(filtered_hl, 1):
            if h["is_spam"]:
                card_class = "line-card-spam"
                tag_badge = "<span class='badge badge-spam'>SPAM</span>"
            elif "High" in h["urgency"]:
                card_class = "line-card-urgent"
                tag_badge = "<span class='badge badge-urgent'>HIGH PRIORITY</span>"
            elif h["is_promo"]:
                card_class = "line-card"
                tag_badge = "<span class='badge badge-promo'>PROMO</span>"
            else:
                card_class = "line-card"
                tag_badge = "<span class='badge badge-clean'>CLEAN</span>"

            st.markdown(f"""
            <div class="{card_class}">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <div>
                        <strong>#{idx}. {h['subject']}</strong>
                        <span style="color:#666; font-size:12px; margin-left:8px;">From: <code>{h['sender']}</code> · {h['date']}</span>
                    </div>
                    <div>
                        {tag_badge}
                        <span class="badge badge-category">{h['category']}</span>
                    </div>
                </div>
                <div style="font-size:14px; color:#1a202c; padding-top:4px;">
                    {h['one_line_important']}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==========================================================================
# VIEW: SPAM & PROMOTIONS ISOLATION
# ==========================================================================
elif nav_choice == "Spam & Promotions":
    st.markdown("## Spam & Promotional Email Isolation")
    st.caption("Automatic detection and quarantine of spam, phishing, scams, and marketing newsletters.")

    if not all_emails:
        st.warning("No emails loaded.")
    else:
        digest = generate_inbox_master_digest(all_emails)
        spam_list = digest["spam_emails"]
        promo_list = digest["promo_emails"]

        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Quarantined spam / scam", len(spam_list))
        col_s2.metric("Newsletters & promotions", len(promo_list))
        col_s3.metric("Safe / important inbox", len(digest["clean_emails"]))

        st.divider()

        tab_spam, tab_promo = st.tabs(["Flagged Spam & Scams", "Marketing & Promotions"])

        with tab_spam:
            if spam_list:
                st.error(f"{len(spam_list)} suspicious message(s) found and kept out of the main inbox.")
                for sp in spam_list:
                    analysis = sp["analysis"]
                    spam_info = analysis["spam_info"]
                    with st.container(border=True):
                        c_sp1, c_sp2 = st.columns([3, 1])
                        with c_sp1:
                            st.markdown(f"### {sp['subject']}")
                            st.caption(f"Sender: `{sp['sender']}` · {sp['date']}")
                            st.markdown(f"**Flagged for:** {', '.join(spam_info['reasons'])}")
                            with st.expander("View raw message content"):
                                mail_full = next((m for m in all_emails if m["id"] == sp["id"]), None)
                                if mail_full:
                                    st.code(mail_full.get("body_plain", "")[:800], language="text")
                        with c_sp2:
                            st.markdown(
                                f"<span class='badge badge-spam'>Spam score: {spam_info['spam_score']}/100</span>",
                                unsafe_allow_html=True,
                            )
                            st.markdown("<span class='badge badge-urgent'>Quarantined</span>", unsafe_allow_html=True)
            else:
                st.success("No spam or phishing emails detected.")

        with tab_promo:
            if promo_list:
                st.info(f"{len(promo_list)} promotional message(s) found.")
                for pr in promo_list:
                    analysis = pr["analysis"]
                    with st.container(border=True):
                        c_pr1, c_pr2 = st.columns([3, 1])
                        with c_pr1:
                            st.markdown(f"**{pr['subject']}**")
                            st.caption(f"Sender: `{pr['sender']}` · {pr['date']}")
                            st.markdown(f"**Summary:** {analysis['summary']}")
                        with c_pr2:
                            st.markdown("<span class='badge badge-promo'>Promotion</span>", unsafe_allow_html=True)
                            st.markdown(f"<span class='badge badge-category'>{analysis['topic']}</span>", unsafe_allow_html=True)
            else:
                st.success("No promotional newsletters found.")


# ==========================================================================
# VIEW: INBOX & DETAIL VIEW
# ==========================================================================
elif nav_choice == "Inbox & Detail View":
    st.markdown("## Inbox & Message Detail")
    st.caption("Browse live Gmail messages with full AI analysis, attachment previews, and the original content.")

    if not all_emails:
        st.warning("No emails loaded.")
    else:
        col_list, col_detail = st.columns([1.1, 1.9])

        with col_list:
            st.markdown(f"### Messages ({len(all_emails)})")
            inbox_filter = st.selectbox(
                "Filter", ["All Mail", "Clean / Important Only", "Unread Only", "Urgent Only", "Has Attachments"]
            )

            filtered_inbox = all_emails
            if inbox_filter == "Clean / Important Only":
                filtered_inbox = [
                    m for m in all_emails
                    if not detect_spam_and_promotions(m["subject"], m.get("body_plain", ""), m.get("sender", ""))["is_spam"]
                ]
            elif inbox_filter == "Unread Only":
                filtered_inbox = [m for m in all_emails if m.get("unread", False)]
            elif inbox_filter == "Urgent Only":
                filtered_inbox = [m for m in all_emails if m.get("is_important", False)]
            elif inbox_filter == "Has Attachments":
                filtered_inbox = [m for m in all_emails if m.get("has_attachments", False)]

            for m in filtered_inbox:
                is_selected = (m["id"] == st.session_state.get("selected_email_id"))
                unread_badge = "● " if m.get("unread", False) else ""
                att_badge = "📎 " if m.get("has_attachments", False) else ""

                sender_name = m.get("sender", "Unknown").split("<")[0].strip() or m.get("sender", "Unknown")
                subject_display = m["subject"][:38] + "..." if len(m["subject"]) > 38 else m["subject"]
                btn_label = f"{unread_badge}{att_badge}{subject_display}"

                if st.button(
                    f"**{btn_label}**\n\n`{sender_name[:25]}` · {m.get('date', '')[:10]}",
                    key=f"btn_mail_{m['id']}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary",
                ):
                    st.session_state["selected_email_id"] = m["id"]
                    st.rerun()

        with col_detail:
            selected_mail = next(
                (m for m in all_emails if m["id"] == st.session_state.get("selected_email_id")), all_emails[0]
            )
            analysis = analyze_single_email(selected_mail, gemini_api_key=st.session_state.get("gemini_api_key"))

            st.markdown(f"## {selected_mail.get('subject', 'No Subject')}")

            b_cols = st.columns([1, 1, 1, 2])
            with b_cols[0]:
                st.markdown(f"<span class='badge badge-category'>{analysis['topic']}</span>", unsafe_allow_html=True)
            with b_cols[1]:
                if analysis["spam_info"]["is_spam"]:
                    st.markdown("<span class='badge badge-spam'>SPAM</span>", unsafe_allow_html=True)
                elif analysis["spam_info"]["is_promo"]:
                    st.markdown("<span class='badge badge-promo'>PROMO</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='badge badge-clean'>CLEAN</span>", unsafe_allow_html=True)
            with b_cols[2]:
                st.markdown(f"<span class='badge badge-urgent'>{analysis['urgency']}</span>", unsafe_allow_html=True)

            st.caption(
                f"**From:** `{selected_mail.get('sender', '')}` · "
                f"**To:** `{selected_mail.get('to', '')}` · "
                f"**Date:** {selected_mail.get('date', '')}"
            )

            st.divider()

            with st.container(border=True):
                st.markdown("### AI Briefing")
                st.markdown(f"**Key takeaway:** {analysis['one_line_important']}")
                st.markdown(f"**Summary:** {analysis['summary']}")

                if analysis["key_points"]:
                    st.markdown("**Key points:**")
                    for kp in analysis["key_points"]:
                        st.markdown(f"- {kp}")

                if analysis["action_items"]:
                    st.markdown("**Action items:**")
                    for act in analysis["action_items"]:
                        st.markdown(f"- `{act}`")

            if selected_mail.get("has_attachments"):
                st.markdown("### Attachments")
                for att in selected_mail.get("attachments", []):
                    with st.container(border=True):
                        st.markdown(f"**{att['filename']}** ({round(att.get('size', 0) / 1024, 1)} KB)")
                        if att.get("extracted_text"):
                            with st.expander("View extracted document content"):
                                st.text(att["extracted_text"][:2000])

            st.markdown("### Original Message")
            tab_plain, tab_html = st.tabs(["Plain Text", "Rich View"])
            with tab_plain:
                st.text_area("Body", selected_mail.get("body_plain", ""), height=250, disabled=True)
            with tab_html:
                if selected_mail.get("body_html"):
                    st.components.v1.html(selected_mail.get("body_html", ""), height=300, scrolling=True)
                else:
                    st.info("No HTML version available for this message.")


# ==========================================================================
# VIEW: URGENT & ACTION REQUIRED
# ==========================================================================
elif nav_choice == "Urgent & Action Required":
    st.markdown("## Urgent & Action Required")
    st.caption("Messages that need a prompt response, a completed task, or have an approaching deadline.")

    if not all_emails:
        st.warning("No emails loaded.")
    else:
        digest = generate_inbox_master_digest(all_emails)
        urgent_list = digest["urgent_emails"]

        if urgent_list:
            st.markdown(f"**{len(urgent_list)}** item(s) require attention:")
            for idx, u in enumerate(urgent_list, 1):
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"### {idx}. {u['subject']}")
                        st.caption(f"From: `{u['sender']}` · {u['date']}")
                        st.markdown(f"**Required action:** {u['action']}")
                    with c2:
                        st.markdown(f"<span class='badge badge-urgent'>{u['urgency']}</span>", unsafe_allow_html=True)
                        st.markdown(f"<span class='badge badge-category'>{u['topic']}</span>", unsafe_allow_html=True)
        else:
            st.success("No urgent actions currently pending.")


# ==========================================================================
# VIEW: AI REPLY COMPOSER
# ==========================================================================
elif nav_choice == "AI Reply Composer":
    st.markdown("## AI Reply Composer")
    st.caption("Draft a professional, casual, or short reply and send it directly through Gmail.")

    if not all_emails:
        st.warning("No emails loaded.")
    else:
        col_r1, col_r2 = st.columns([1, 1.2])

        with col_r1:
            st.markdown("### Replying To")
            email_options = {f"{m['subject'][:40]}... ({m.get('sender', '')[:20]})": m["id"] for m in all_emails}
            selected_title = st.selectbox("Select a message", list(email_options.keys()), index=0)
            sel_id = email_options[selected_title]
            current_target = next(m for m in all_emails if m["id"] == sel_id)

            with st.container(border=True):
                st.markdown(f"**Subject:** {current_target['subject']}")
                st.markdown(f"**From:** `{current_target['sender']}`")
                st.caption(f"**Date:** {current_target['date']}")
                st.text_area("Snippet", current_target.get("body_plain", "")[:400], height=150, disabled=True)

            reply_tone = st.selectbox(
                "Tone", ["Professional", "Friendly & Casual", "Short & Direct", "Polite Decline", "Request More Info"]
            )
            custom_notes = st.text_input(
                "Notes to include", placeholder="e.g. Available Friday at 3 PM, thanks."
            )

            if st.button("Generate Draft", type="primary", use_container_width=True):
                with st.spinner("Drafting response..."):
                    draft = generate_ai_reply(
                        current_target,
                        tone=reply_tone,
                        custom_instructions=custom_notes,
                        gemini_api_key=st.session_state.get("gemini_api_key"),
                    )
                    st.session_state["reply_draft"] = draft
                st.success("Draft ready.")

        with col_r2:
            st.markdown("### Reply Draft")
            draft_text = st.text_area("Message", value=st.session_state.get("reply_draft", ""), height=300)
            st.session_state["reply_draft"] = draft_text

            col_send1, col_send2 = st.columns(2)
            with col_send1:
                if st.button(
                    "Send via Gmail",
                    use_container_width=True,
                    type="primary",
                    disabled=not st.session_state["gmail_connected"],
                ):
                    if not draft_text.strip():
                        st.error("The reply is empty.")
                    else:
                        with st.spinner("Sending via Gmail SMTP..."):
                            client = GmailClient(st.session_state["gmail_user"], st.session_state["gmail_pass"])
                            sender_email_match = re.search(r"<([^>]+)>", current_target["sender"])
                            target_to = sender_email_match.group(1) if sender_email_match else current_target["sender"]

                            ok, send_msg = client.send_reply(
                                to_email=target_to,
                                subject=current_target["subject"],
                                body_text=draft_text,
                                in_reply_to=current_target.get("message_id"),
                            )
                            if ok:
                                st.success(send_msg)
                            else:
                                st.error(send_msg)
            with col_send2:
                if not st.session_state["gmail_connected"]:
                    st.caption("Connect a live Gmail account in the sidebar to send real replies.")


# ==========================================================================
# VIEW: FILE IMPORTER (JSON / CSV / PDF)
# ==========================================================================
elif nav_choice == "Import Files (JSON/CSV/PDF)":
    st.markdown("## File Importer")
    st.caption("Load local JSON, CSV, TXT, or PDF files for analysis — nothing is stored in a database.")

    uploaded_files = st.file_uploader(
        "Drop email files or datasets here", type=["json", "csv", "txt", "pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("Process & Load Files", type="primary"):
            imported: List[Dict[str, Any]] = []
            for idx, f in enumerate(uploaded_files, 1):
                fname = f.name
                fbytes = f.read()

                if fname.endswith(".json"):
                    try:
                        data = json.loads(fbytes.decode("utf-8", errors="replace"))
                        if isinstance(data, list):
                            imported.extend(data)
                        elif isinstance(data, dict):
                            imported.append(data)
                    except Exception as e:
                        st.error(f"Could not parse {fname}: {e}")
                elif fname.endswith(".pdf"):
                    pdf_text = extract_pdf_text_from_bytes(fbytes)
                    imported.append({
                        "id": f"pdf_{idx}",
                        "sender": "Imported PDF Document",
                        "to": "you@domain.com",
                        "subject": f"PDF Import: {fname}",
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "body_plain": pdf_text or "This PDF contained scanned images or unreadable text.",
                        "has_attachments": True,
                        "attachments": [{"filename": fname, "size": len(fbytes), "extracted_text": pdf_text}],
                        "unread": False,
                    })
                elif fname.endswith(".csv"):
                    try:
                        df = pd.read_csv(io.BytesIO(fbytes))
                        for r_idx, row in df.iterrows():
                            imported.append({
                                "id": f"csv_{idx}_{r_idx}",
                                "sender": str(row.get("sender", "Unknown")),
                                "to": str(row.get("to", "")),
                                "subject": str(row.get("subject", "No Subject")),
                                "date": str(row.get("date", datetime.now().strftime("%Y-%m-%d"))),
                                "body_plain": str(row.get("body", row.get("text", ""))),
                                "has_attachments": False,
                                "attachments": [],
                                "unread": False,
                            })
                    except Exception as e:
                        st.error(f"Could not parse {fname}: {e}")

            if imported:
                st.session_state["emails"] = imported
                st.session_state["selected_email_id"] = imported[0]["id"]
                st.success(f"Loaded {len(imported)} message(s) into the current session.")
                st.rerun()


# ==========================================================================
# VIEW: SETTINGS
# ==========================================================================
elif nav_choice == "Settings":
    st.markdown("## Settings")
    st.caption("Configure the AI engine, API keys, and review the app's privacy model.")

    with st.container(border=True):
        st.markdown("### AI Engine")
        st.markdown("""
        - **Built-in NLP engine (default):** runs entirely locally, with no latency or API
          cost, extracting categories, spam scores, action items, dates, and highlights.
        - **Google Gemini (optional):** add a Gemini API key below to enable higher-level
          generative reasoning for summaries and replies.
        """)

        gemini_input = st.text_input(
            "Google Gemini API key (optional)",
            value=st.session_state.get("gemini_api_key", ""),
            type="password",
            placeholder="AIzaSy...",
        )
        if st.button("Save API Key"):
            st.session_state["gemini_api_key"] = gemini_input.strip()
            st.success("API key saved for this session.")

    with st.container(border=True):
        st.markdown("### Privacy & Architecture")
        st.markdown("""
        - **Zero database persistence:** no emails, credentials, or summaries are written to
          disk or to an external database.
        - **Direct, encrypted connection:** all traffic goes directly between this machine and
          `imap.gmail.com:993` / `smtp.gmail.com:587`.
        """)