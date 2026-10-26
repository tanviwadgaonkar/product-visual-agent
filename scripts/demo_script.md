# Demo video — 55 seconds

Record in Chrome at 1440×900 with the page filling the window. Screen-record with your mic on (Loom, OBS or QuickTime). One take is fine; cut pauses. No background music.

## Before recording
- `uvicorn backend.main:app --port 8000` running, Ollama warm (ask one question off-camera so the model is loaded).
- Delete `data/app.db` so the counter starts at 0 demos.
- Check the browser voice sounds natural (Google en-IN or en-GB). Close other tabs.

## Shot list

| Time | On screen | You say (voiceover, calm and quick) |
|---|---|---|
| 0–5 s | Page loaded, avatar idle | "This is a product visual agent I built in two days — a character that answers from a brand's catalogue, shows the product, and takes actions. All local: FastAPI, ChromaDB, Ollama." |
| 5–17 s | Click mic. Ask aloud: "Which colours does the Runner Pro come in?" Avatar speaks; product card + sources appear | (let the agent's voice play) then: "Every answer is grounded — you can see the exact passage it used on the right." |
| 17–28 s | Ask: "Do you have something for trail running?" Card switches to Trail Grip | "Retrieval picks the product, the card follows the conversation." |
| 28–40 s | Ask: "Book me a demo — I'm Asha, asha at example dot com." Chip "Demo booked" appears, counter goes to 1 | "And it acts. The model returns a typed action, the server writes the lead — same pattern as an enterprise sales or support agent." |
| 40–50 s | Cut to VS Code: `agent.py` system prompt, then `main.py` action block (2 s each) | "The whole thing is about 400 lines. Strict JSON from the model, one retrieval step, one action executor." |
| 50–55 s | Back to page, avatar idle. Title card: your name · GitHub link | "Code's on GitHub. Thanks for watching." |

## Export
- 1080p, MP4, under 60 s.
- Make a 6–8 s GIF of the demo-booking moment for the README (`docs/demo.gif`).
- Post on LinkedIn: 2 lines + video + repo link. Tag the company you're pitching.
