"""Generate, email, and remember a grounded daily news briefing.

Required environment variables:
    API_KEY, SENDER_EMAIL, PASSWORD_EMAIL, RECIPIENT_EMAIL
"""

from __future__ import annotations

import html
import json
import os
import smtplib
from datetime import date, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from google import genai
from google.genai import types


MODEL_NAME = "gemini-2.5-flash"
HISTORY_PATH = Path(__file__).resolve().with_name("history.json")
EASTERN_TIME = ZoneInfo("America/New_York")


EDITORIAL_INSTRUCTIONS = r"""
Create a concise, factual daily briefing containing up to ten important recent news stories. Use fewer than ten whenever there are not ten stories that independently meet the relevance and significance threshold; never add filler to reach a numerical target. Rank selected stories strictly by the magnitude of the change, overall significance, and likely lasting impact on the United States. Make the first item the day's most groundbreaking development with the greatest probable long-term effect on the United States.
Use the following as separate editorial categories for classification, not as quotas. Assign every selected story to exactly one category and label it clearly. A story that does not genuinely fit one of these categories must be excluded. Do not force coverage of every category, reserve slots by category, or include a weaker story merely to achieve balance or story count. Select only the strongest genuinely new developments across all categories.
Apply the US-relevance test by category:
\- Foreign wars and armed conflicts do not need a US connection. Select them based on the magnitude or novelty of the military, political, territorial, diplomatic, or humanitarian change, including important developments in small or underreported conflicts.
\- AI and technology stories do not need a US connection, but keep the existing geographic and editorial priorities in that category: focus on consequential US developments and Chinese developments relevant to competition with the US, plus genuinely culture-shaping events.
\- Türkiye stories do not need a US connection. Select consequential Turkish geopolitical, diplomatic, security, defense, regional, or foreign-policy developments even when their effect on Washington is limited.
\- All other stories—including science and space, non-conflict international politics, cybersecurity, China's digital influence in Africa, and uncategorized developments—must have a concrete and substantial effect on US security, policy, diplomacy, alliances, military posture, critical infrastructure, supply chains, trade, technology leadership, energy security, scientific advantage, or another durable national interest. An indirect, speculative, or second-order US connection is insufficient. If the direct US impact cannot be stated convincingly in one sentence, exclude the story.
Exclude general-interest disasters, weather, resource shortages, domestic emergencies, routine elections, and local political developments unless they qualify as a significant foreign-conflict development or have a clear, substantial US consequence.
Before selecting stories, review the events covered in previous briefings available in the provided history. Maintain continuity across runs:
\- Do not repeat or substantially overlap with a previously reported event.
\- Treat multiple articles about the same underlying development as one event.
\- Revisit an ongoing story only when a substantive new development changes its facts, trajectory, scale, participants, policy response, strategic implications, or credible assessment.
\- When revisiting a story, focus on the new development and explicitly state the delta from the last briefing instead of repeating background.
\- Do not use routine follow-ups, restatements, reactions, small numerical updates, or newly published commentary as justification for repeating a story.
\- Prefer a somewhat older but genuinely unreported and important development over a newer article that merely duplicates prior coverage.
CATEGORIES:
1\. AI AND TECHNOLOGY
2\. SCIENCE AND SPACE
3\. INTERNATIONAL POLITICS AND CONFLICTS
4\. TÜRKIYE
5\. CYBERSECURITY
6\. CHINA'S DIGITAL INFLUENCE IN AFRICA
For every story:
\- Provide a short headline and its single category label.
\- Include one relevant accompanying news image URL with a brief caption and source credit.
\- Give a concise 2–3 sentence factual summary.
\- State exactly what is new or changed.
\- Add a one-sentence "Why it matters" explanation, including US implications where relevant.
\- Link directly to a reliable source.
OUTPUT FORMAT:
Generate clean, responsive HTML with inline CSS styling formatted cleanly for email clients (dark text, clean fonts like Helvetica/system sans-serif, responsive card containers, clean padding).
""".strip()


class HeadlineParser(HTMLParser):
    """Extract story titles from the predictable HTML contract in the prompt."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.headlines: list[str] = []
        self._heading_tag: str | None = None
        self._heading_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        data_headline = attributes.get("data-headline")
        if data_headline:
            self.headlines.append(data_headline)

        # h2 is reserved for story headlines by the prompt. h3 is accepted as a
        # graceful fallback if the model uses a slightly different card hierarchy.
        if tag in {"h2", "h3"} and self._heading_tag is None:
            self._heading_tag = tag
            self._heading_parts = []

    def handle_data(self, data: str) -> None:
        if self._heading_tag is not None:
            self._heading_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == self._heading_tag:
            self.headlines.append("".join(self._heading_parts))
            self._heading_tag = None
            self._heading_parts = []


def now_in_eastern() -> datetime:
    return datetime.now(EASTERN_TIME)


def write_history(history: dict[str, list[dict[str, str]]]) -> None:
    """Write atomically so an interrupted run cannot corrupt the history file."""
    temporary_path = HISTORY_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(HISTORY_PATH)


def load_history() -> dict[str, list[dict[str, str]]]:
    """Load history.json, creating the requested empty structure when absent."""
    if not HISTORY_PATH.exists():
        history: dict[str, list[dict[str, str]]] = {"covered_stories": []}
        write_history(history)
        return history

    try:
        raw_history: Any = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{HISTORY_PATH} contains invalid JSON.") from exc

    if not isinstance(raw_history, dict) or not isinstance(raw_history.get("covered_stories"), list):
        raise RuntimeError(
            f"{HISTORY_PATH} must contain an object with a 'covered_stories' list."
        )

    # Keep only well-formed records. A malformed old record must not make the
    # scheduled job fail or pollute the deduplication prompt.
    stories: list[dict[str, str]] = []
    for record in raw_history["covered_stories"]:
        if (
            isinstance(record, dict)
            and isinstance(record.get("date"), str)
            and isinstance(record.get("headline"), str)
            and record["headline"].strip()
        ):
            stories.append(
                {"date": record["date"], "headline": record["headline"].strip()}
            )
    return {"covered_stories": stories}


def parse_record_date(record: dict[str, str]) -> date | None:
    try:
        return date.fromisoformat(record["date"])
    except ValueError:
        return None


def covered_headlines_from_last_seven_days(
    history: dict[str, list[dict[str, str]]], today: date
) -> list[str]:
    cutoff = today - timedelta(days=6)
    return [
        story["headline"]
        for story in history["covered_stories"]
        if (story_date := parse_record_date(story)) is not None and story_date >= cutoff
    ]


def format_previous_headlines(headlines: list[str]) -> str:
    if not headlines:
        return "- None — this is the first briefing."
    return "\n".join(f"- {headline}" for headline in headlines)


def build_prompt(previous_headlines: list[str]) -> str:
    return f"""### PREVIOUSLY COVERED HEADLINES (DO NOT REPEAT)
{format_previous_headlines(previous_headlines)}

{EDITORIAL_INSTRUCTIONS}

HTML EXTRACTION CONTRACT:
Return only the complete HTML email document—no Markdown fences or explanatory text.
Use exactly one `<article class="story-card" data-headline="SHORT HEADLINE">` for each selected story, with the same short headline in a descendant `<h2>`. The `data-headline` value must be plain text and must not contain HTML. This enables the next briefing to avoid repeating the event.
"""


def generate_briefing_html(prompt: str) -> str:
    api_key = os.environ["API_KEY"]
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[{"google_search": {}}],
            temperature=0.2,
        ),
    )
    if not response.text or not response.text.strip():
        raise RuntimeError("Gemini returned an empty briefing.")
    return response.text.strip()


def normalize_headline(headline: str) -> str:
    return " ".join(html.unescape(headline).split()).casefold()


def extract_headlines(email_html: str) -> list[str]:
    parser = HeadlineParser()
    parser.feed(email_html)
    parser.close()

    unique_headlines: list[str] = []
    seen: set[str] = set()
    for headline in parser.headlines:
        clean_headline = " ".join(html.unescape(headline).split())
        normalized = normalize_headline(clean_headline)
        if clean_headline and normalized not in seen:
            seen.add(normalized)
            unique_headlines.append(clean_headline)
    return unique_headlines


def update_history(
    history: dict[str, list[dict[str, str]]], headlines: list[str], today: date
) -> dict[str, list[dict[str, str]]]:
    """Append this run's new titles and retain only the most recent 14 days."""
    cutoff = today - timedelta(days=13)
    retained = [
        story
        for story in history["covered_stories"]
        if (story_date := parse_record_date(story)) is not None and story_date >= cutoff
    ]
    known_titles = {normalize_headline(story["headline"]) for story in retained}
    for headline in headlines:
        normalized = normalize_headline(headline)
        if normalized and normalized not in known_titles:
            retained.append({"date": today.isoformat(), "headline": headline})
            known_titles.add(normalized)
    return {"covered_stories": retained}


def send_email(email_html: str, briefing_date: datetime) -> None:
    sender = os.environ["SENDER_EMAIL"]
    password = os.environ["PASSWORD_EMAIL"]
    recipient = os.environ["RECIPIENT_EMAIL"]

    message = MIMEMultipart("alternative")
    message["From"] = sender
    message["To"] = recipient
    formatted_date = f"{briefing_date.strftime('%B')} {briefing_date.day}, {briefing_date.year}"
    message["Subject"] = f"Daily News Briefing — {formatted_date}"
    message.attach(MIMEText(email_html, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, [recipient], message.as_string())


def main() -> None:
    current_time = now_in_eastern()
    today = current_time.date()
    history = load_history()
    previous_headlines = covered_headlines_from_last_seven_days(history, today)
    email_html = generate_briefing_html(build_prompt(previous_headlines))
    new_headlines = extract_headlines(email_html)

    # Only mark stories covered after Gmail accepts the message. A transient email
    # failure should not suppress those stories in tomorrow's run.
    send_email(email_html, current_time)
    write_history(update_history(history, new_headlines, today))
    print(f"Sent briefing with {len(new_headlines)} tracked headline(s).")


if __name__ == "__main__":
    main()
