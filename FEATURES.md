# EMO local feature catalog

The local feature layer is in `feature_hub.py`. It runs before LM Studio fallback and requires no additional pip packages.

## Examples by category

- **Calendar:** current date, weekday, year, month, ISO week, day-of-year, leap year, days until a date, date difference, weekday for a date, days since a date, next weekday.
- **Math:** safe calculator, square, cube, square root, power, percentage, percentage change, sum, average/mean, median, minimum, maximum, factorial, prime check/list, Fibonacci sequence, GCD, LCM, rounding, absolute value, floor, ceiling, fraction-to-decimal.
- **Conversions:** Celsius/Fahrenheit/Kelvin; distance, mass, volume, speed, time, and decimal/binary data-size units.
- **Text:** word/character/line counts; upper/lower/title case; reverse; palindrome; sort/unique words; word frequency; vowel/consonant counts; collapse spaces; strip punctuation; slugify; initials/acronym; repeat text; Unicode character codes; text replacement; trim.
- **Encodings:** base64, URL encoding, binary bytes, hexadecimal bytes, JSON pretty-print and compact JSON.
- **Random/fun:** random integer, coin, dice, choice from options, 16-character random password, UUID, offline joke, riddle, trivia fact, positive encouragement.
- **Organizer:** add/list/search/delete notes; add/list/complete/delete tasks; clear completed tasks; countdown timer; cancel timer; timer status; 25-minute focus timer.
- **Diagnostics:** device summary, RAM, disk, uptime, CPU temperature when the OS exposes it, local IP, local interface status, OS, Python, process/thread count.
- **Face:** explicit happy, sad, excited, sleepy, dance, idle/reset, and thinking expressions, plus help/catalog discovery.

## Notes and limitations

Use ISO dates (`YYYY-MM-DD`) for date calculations. Unit conversion accepts abbreviations such as `convert 5 km to miles`, `convert 2 kg to lb`, `convert 90 minutes to hours`, and `100 c to f`. Text commands use explicit prefixes such as `uppercase: hello world` to avoid accidentally transforming ordinary conversation.

Notes and tasks are stored in the local SQLite database under `data/`. Countdown/Pomodoro timers run only while EMO is running. No feature evaluates arbitrary Python or invokes a shell command. Internet status is not inferred from the presence of a local network address.
