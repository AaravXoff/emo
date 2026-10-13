"""Deterministic local features for EMO; no extra runtime packages are required."""
from __future__ import annotations
import ast, base64, datetime as dt, json, math, os, platform, random, re
import shutil, socket, sqlite3, statistics, string, threading, time, urllib.parse, uuid
from pathlib import Path
from typing import Callable

FEATURES = (
    # Calendar/date: 12
    "current_date", "weekday", "year", "month", "week_number", "day_of_year",
    "leap_year", "days_until", "date_difference", "weekday_for_date", "days_from_date", "next_weekday",
    # Math/number: 22
    "calculator", "square", "cube", "square_root", "power", "percentage", "percentage_change",
    "sum", "average", "median", "minimum", "maximum", "factorial", "prime_check", "prime_list",
    "fibonacci", "greatest_common_divisor", "least_common_multiple", "round_number", "absolute_value",
    "floor_number", "ceiling_number", "fraction_to_decimal",
    # Conversion: 10
    "temperature_conversion", "length_conversion", "mass_conversion", "volume_conversion",
    "speed_conversion", "time_conversion", "data_conversion", "distance_miles_km", "weight_kg_lb", "temperature_c_f",
    # Text toolbox: 22
    "word_count", "character_count", "line_count", "uppercase", "lowercase", "title_case",
    "reverse_text", "palindrome_check", "sort_words", "unique_words", "word_frequency", "vowel_count",
    "consonant_count", "collapse_spaces", "remove_punctuation", "slugify", "initials", "acronym",
    "repeat_text", "character_codes", "replace_text", "text_trim",
    # Encoding: 10
    "base64_encode", "base64_decode", "url_encode", "url_decode", "binary_encode", "binary_decode",
    "hex_encode", "hex_decode", "json_pretty", "json_compact",
    # Random/play: 10
    "random_number", "coin_flip", "dice_roll", "random_choice", "password_generator", "uuid_generator",
    "joke", "riddle", "trivia", "compliment",
    # Organizer: 13
    "note_add", "note_list", "note_search", "note_delete", "todo_add", "todo_list", "todo_complete",
    "todo_delete", "todo_clear_completed", "countdown_timer", "cancel_timer", "pomodoro_timer", "timer_status",
    # Device diagnostics: 10
    "device_summary", "memory_status", "disk_status", "uptime_status", "cpu_temperature", "ip_address",
    "network_status", "os_version", "python_version", "process_status",
    # Face/assistant: 8
    "face_happy", "face_sad", "face_excited", "face_sleepy", "face_dance", "face_idle", "face_thinking", "help_catalog",
)
assert len(FEATURES) >= 100


class SafeMath:
    """Restricted AST evaluator: arithmetic only, never eval() or arbitrary Python."""
    BINARY = {ast.Add: lambda a,b:a+b, ast.Sub:lambda a,b:a-b, ast.Mult:lambda a,b:a*b,
              ast.Div:lambda a,b:a/b, ast.FloorDiv:lambda a,b:a//b, ast.Mod:lambda a,b:a%b,
              ast.Pow:lambda a,b:a**b}
    UNARY = {ast.UAdd:lambda x:+x, ast.USub:lambda x:-x}
    FUNCTIONS = {"sqrt":math.sqrt,"abs":abs,"round":round,"floor":math.floor,
                 "ceil":math.ceil,"sin":math.sin,"cos":math.cos,"tan":math.tan,
                 "log":math.log10,"ln":math.log,"factorial":math.factorial}

    @classmethod
    def evaluate(cls, expression):
        expression = expression.strip().replace("^", "**")
        if not expression or len(expression) > 160: raise ValueError("Please give me a short expression.")
        tree = ast.parse(expression, mode="eval")
        def visit(node):
            if isinstance(node, ast.Expression): return visit(node.body)
            if isinstance(node, ast.Constant) and type(node.value) in (int,float):
                if abs(node.value) > 10**15: raise ValueError("That number is too large.")
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in cls.BINARY:
                a,b = visit(node.left),visit(node.right)
                if isinstance(node.op, ast.Pow) and abs(b)>12: raise ValueError("Exponent must be between -12 and 12.")
                out=cls.BINARY[type(node.op)](a,b)
                if isinstance(out,complex) or not math.isfinite(float(out)) or abs(out)>1e100:
                    raise ValueError("That result is outside the supported range.")
                return out
            if isinstance(node, ast.UnaryOp) and type(node.op) in cls.UNARY: return cls.UNARY[type(node.op)](visit(node.operand))
            if isinstance(node, ast.Call) and isinstance(node.func,ast.Name) and node.func.id in cls.FUNCTIONS:
                if node.keywords or len(node.args) not in (1,2): raise ValueError("Unsupported function arguments.")
                return cls.FUNCTIONS[node.func.id](*(visit(x) for x in node.args))
            raise ValueError("I can calculate numeric expressions and supported math functions only.")
        return visit(tree)


def is_leap(year): return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
def is_prime(n):
    if n<2:return False
    if n in (2,3):return True
    if n%2==0 or n%3==0:return False
    d=5
    while d*d<=n:
        if n%d==0 or n%(d+2)==0:return False
        d+=6
    return True
def unit_key(v):
    v=v.strip().lower().replace("°","")
    return {"celsius":"c","fahrenheit":"f","kelvin":"k","kilometers":"km","kilometres":"km",
      "kilometer":"km","kilometre":"km","meters":"m","metres":"m","meter":"m","metre":"m",
      "pounds":"lbs","pound":"lb","kilograms":"kg","kilogram":"kg","grams":"g","gram":"g",
      "liters":"liter","litres":"litre","milliliters":"ml","millilitres":"ml"}.get(v,v)
def unique_words(text):
    seen=set(); out=[]
    for word in text.split():
        if word.casefold() not in seen: seen.add(word.casefold()); out.append(word)
    return " ".join(out)
def slugify(text):
    return re.sub(r"[-\s]+","-",re.sub(r"[^\w\s-]","",text.casefold(),flags=re.UNICODE)).strip("-")


class LocalFeatureHub:
    """100+ deterministic commands; persistence is local to data/emo_features.sqlite3."""
    def __init__(self,data_dir:Path,status=None,face=None,hardware=None,notify:Callable[[str],None]|None=None):
        self.data_dir=Path(data_dir); self.data_dir.mkdir(parents=True,exist_ok=True)
        self.status,self.face,self.hardware,self.notify=status,face,hardware,notify
        self.lock=threading.RLock(); self.timers={}
        self.db=sqlite3.connect(str(self.data_dir/"emo_features.sqlite3"),check_same_thread=False,timeout=10)
        self.db.row_factory=sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL"); self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY,created_at TEXT NOT NULL,text TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS todos(id INTEGER PRIMARY KEY,created_at TEXT NOT NULL,text TEXT NOT NULL,
          done INTEGER NOT NULL DEFAULT 0,completed_at TEXT);
        """); self.db.commit()

    @property
    def feature_count(self): return len(FEATURES)
    def catalog(self): return list(FEATURES)
    @staticmethod
    def fmt(x): return f"{x:.10g}" if isinstance(x,float) and math.isfinite(x) else str(x)
    @staticmethod
    def now_text(): return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    @staticmethod
    def numbers(s): return [float(v) for v in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",s)]

    def handle(self,text:str):
        original=(text or "").strip(); t=original.casefold()
        if not original:return None
        if t in {"help","what can you do","what features do you have","list your features","show features","show all features","what commands do you know"}:
            return "help_catalog",(f"I have {self.feature_count} lightweight local features: dates, math, unit conversion, text tools, encoders, random utilities, notes, to-dos, timers, device diagnostics, and face controls. Try 'calculate 12 * (3 + 4)', 'convert 5 km to miles', 'count words in: hello there', 'add note: remember this', 'add todo: feed the cat', or 'show device status'.")

        # Persistent notes and tasks.
        m=re.match(r"^(?:add|save|remember)(?: a)? note\s*[:,-]?\s+(.+)$",original,re.I)
        if m:
            value=m.group(1).strip()[:2000]
            with self.lock,self.db: cur=self.db.execute("INSERT INTO notes(created_at,text) VALUES(?,?)",(self.now_text(),value))
            return "note_add",f"Saved note {cur.lastrowid}: {value}"
        if t in {"list notes","show notes","read my notes","what are my notes"}:
            with self.lock: rows=self.db.execute("SELECT id,text FROM notes ORDER BY id DESC LIMIT 20").fetchall()
            return "note_list",("Your recent notes: "+"; ".join(f"{r['id']}: {r['text']}" for r in rows)) if rows else "No saved notes yet. Say: add note: your note."
        m=re.match(r"^(?:search|find) notes?(?: for)?\s+(.+)$",original,re.I)
        if m:
            with self.lock: rows=self.db.execute("SELECT id,text FROM notes WHERE text LIKE ? ORDER BY id DESC LIMIT 10",(f"%{m.group(1)[:100]}%",)).fetchall()
            return "note_search","; ".join(f"{r['id']}: {r['text']}" for r in rows) if rows else "No matching notes found."
        m=re.match(r"^(?:delete|remove) note\s+#?(\d+)$",t)
        if m:
            with self.lock,self.db: cur=self.db.execute("DELETE FROM notes WHERE id=?",(int(m.group(1)),))
            return "note_delete","Deleted that note." if cur.rowcount else "I couldn't find that note ID."
        m=re.match(r"^(?:add|create)(?: a)? todo\s*[:,-]?\s+(.+)$",original,re.I)
        if m:
            value=m.group(1).strip()[:1000]
            with self.lock,self.db: cur=self.db.execute("INSERT INTO todos(created_at,text) VALUES(?,?)",(self.now_text(),value))
            return "todo_add",f"Added task {cur.lastrowid}: {value}"
        if t in {"list todos","show todos","show my tasks","list tasks","what are my tasks"}:
            with self.lock: rows=self.db.execute("SELECT id,text,done FROM todos ORDER BY done,id DESC LIMIT 30").fetchall()
            return "todo_list","Your tasks: "+"; ".join(f"{'done' if r['done'] else 'pending'} {r['id']}: {r['text']}" for r in rows) if rows else "Your task list is empty. Say: add todo: your task."
        m=re.match(r"^(?:complete|finish|check off) todo\s+#?(\d+)$",t)
        if m:
            with self.lock,self.db: cur=self.db.execute("UPDATE todos SET done=1,completed_at=? WHERE id=?",(self.now_text(),int(m.group(1))))
            return "todo_complete","Task marked complete." if cur.rowcount else "I couldn't find that task ID."
        m=re.match(r"^(?:delete|remove) todo\s+#?(\d+)$",t)
        if m:
            with self.lock,self.db: cur=self.db.execute("DELETE FROM todos WHERE id=?",(int(m.group(1)),))
            return "todo_delete","Removed that task." if cur.rowcount else "I couldn't find that task ID."
        if t in {"clear completed todos","clear completed tasks","remove completed todos"}:
            with self.lock,self.db: cur=self.db.execute("DELETE FROM todos WHERE done=1")
            return "todo_clear_completed",f"Cleared {cur.rowcount} completed task(s)."

        # Countdown timers are process-local; they are cancelled cleanly when EMO exits.
        m=re.match(r"^(?:set )?(?:a )?timer(?: for)?\s+(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)(?:\s+(?:to|for)\s+(.+))?$",t)
        if not m: m=re.match(r"^remind me in\s+(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\s+(?:to\s+)?(.+)$",original,re.I)
        if m:
            amount=float(m.group(1)); unit=m.group(2).lower()
            seconds=amount*(3600 if unit.startswith(("h","hr")) else 60 if unit.startswith(("m","min")) else 1)
            if seconds<=0 or seconds>7*86400:return "countdown_timer","Timer length must be greater than zero and no more than seven days."
            label=(m.group(3) or "your timer is finished").strip()[:180]; tid=uuid.uuid4().hex[:8]
            def fire():
                with self.lock:self.timers.pop(tid,None)
                message=f"Timer finished: {label}."
                if self.notify:self.notify(message)
                if self.status:self.status.set_stage("Timer finished",label)
            timer=threading.Timer(seconds,fire); timer.daemon=True
            with self.lock:self.timers[tid]=timer
            timer.start()
            return "countdown_timer",f"Timer {tid} started for {self.fmt(amount)} {unit}. I'll notify you when it finishes."
        m=re.match(r"^(?:cancel|stop) timer(?:\s+([a-f0-9]{4,12}))?$",t)
        if m:
            tid=m.group(1)
            with self.lock:
                if tid:
                    timer=self.timers.pop(tid,None); timers=[timer] if timer else []
                else: timers=list(self.timers.values()); self.timers.clear()
            for timer in timers:timer.cancel()
            return "cancel_timer",f"Cancelled {len(timers)} timer(s)." if timers else "There are no active timers."
        if t in {"timer status","list timers","how many timers"}:
            with self.lock:n=len(self.timers)
            return "timer_status",f"There {'is' if n==1 else 'are'} {n} active timer(s)."
        if t in {"start pomodoro","start a pomodoro","start focus timer"}:
            self._start_timer(25*60,"your 25-minute focus session is complete")
            return "pomodoro_timer","Your 25-minute focus timer has started. Take a short break when it finishes."

        # Explicit face controls. Casual statements do not change expressions.
        faces={"show happy face":("happy","Happy face is on."),"be happy":("happy","Happy face!"),
          "show sad face":("sad","Sad face is on."),"be sad":("sad","Sad face."),
          "show excited face":("excited","Excited face is on."),"get excited":("excited","Excited face!"),
          "show sleepy face":("sleepy","Sleepy face is on."),"go to sleep":("sleepy","Going into sleepy mode."),
          "dance":("dance","Let's dance!"),"dance for me":("dance","Let's dance!"),
          "show thinking face":("thinking","Thinking face is on."),"reset face":("idle","Face reset."),
          "look at me":("happy","Hi!")}
        if t in faces:
            state,reply=faces[t]
            if self.face:self.face.set_state(state,duration=3 if state!="idle" else .2)
            if self.hardware:self.hardware.set_expression(state)
            return "face_"+state,reply

        # Device diagnostics: local reads only; no shell calls and no external network request.
        if t in {"device status","show device status","system status","how is the system","robot status","how are you running"}:
            temp=self._cpu_temp(); temp_text=f"{temp:.1f} C" if temp is not None else "not reported"
            return "device_summary",f"EMO runs on {platform.system()} {platform.release()} with Python {platform.python_version()}. CPU temperature: {temp_text}. {self._memory()}"
        if t in {"memory status","check memory","how much memory is free","ram status"}:return "memory_status",self._memory()
        if t in {"disk status","disk space","how much disk space is free","storage status"}:
            d=shutil.disk_usage(self.data_dir);return "disk_status",f"Disk space: {d.free/1024**3:.2f} GB free of {d.total/1024**3:.2f} GB."
        if t in {"uptime","system uptime","how long have you been running"}:
            try:s=float(Path("/proc/uptime").read_text().split()[0])
            except Exception:return "uptime_status","Uptime information is unavailable on this operating system."
            return "uptime_status",f"System uptime is {self.duration(s)}."
        if t in {"cpu temperature","processor temperature","how hot is the pi","check cpu temperature"}:
            v=self._cpu_temp();return "cpu_temperature",f"CPU temperature is {v:.1f} degrees Celsius." if v is not None else "CPU temperature is unavailable."
        if t in {"my ip address","what is my ip address","local ip address","network ip"}:return "ip_address","Local IP address: "+self._local_ip()
        if t in {"network status","check network","am i connected to the internet"}:
            ip=self._local_ip();return "network_status",("A local network interface is active at "+ip+". Internet access was not tested." if ip!="unavailable" else "No active local network interface was detected.")
        if t in {"os version","operating system","what operating system are you running"}:return "os_version",f"Operating system: {platform.platform()}."
        if t in {"python version","what python version","python runtime version"}:return "python_version",f"Python runtime: {platform.python_version()}."
        if t in {"process status","what process am i running in","runtime status"}:return "process_status",f"EMO process ID is {os.getpid()}; Python thread count is {threading.active_count()}."

        # Calendar/date commands.
        now=dt.datetime.now().astimezone()
        if t in {"what date is it","what is today's date","today's date","current date","date today"}:return "current_date","Today is "+now.strftime("%A, %B %d, %Y")+"."
        if t in {"what day is it","what day of the week is it","day today","what weekday is it"}:return "weekday","Today is "+now.strftime("%A")+"."
        if t in {"what year is it","current year"}:return "year",f"The current year is {now.year}."
        if t in {"what month is it","current month"}:return "month",f"It's {now.strftime('%B')}."
        if t in {"what week of the year is it","week number"}:return "week_number",f"This is week {now.isocalendar().week} of the year."
        if t in {"what day of the year is it","day of the year"}:return "day_of_year",f"Today is day {now.timetuple().tm_yday} of the year."
        if t in {"is this a leap year","is it a leap year","leap year"}:return "leap_year",f"{now.year} is {'a leap year' if is_leap(now.year) else 'not a leap year'}."
        m=re.match(r"^(?:days until|how many days until)\s+(\d{4}-\d{2}-\d{2})$",t)
        if m:
            target=dt.date.fromisoformat(m.group(1));days=(target-now.date()).days
            return "days_until",f"{target.isoformat()} is {abs(days)} day(s) {'from now' if days>=0 else 'in the past'}."
        m=re.match(r"^days between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})$",t)
        if m:
            a,b=dt.date.fromisoformat(m.group(1)),dt.date.fromisoformat(m.group(2))
            return "date_difference",f"There are {abs((b-a).days)} days between those dates."
        m=re.match(r"^(?:what day was|weekday for)\s+(\d{4}-\d{2}-\d{2})$",t)
        if m:
            d=dt.date.fromisoformat(m.group(1));return "weekday_for_date",f"{d.isoformat()} falls on a {d.strftime('%A')}."
        m=re.match(r"^days since\s+(\d{4}-\d{2}-\d{2})$",t)
        if m:
            d=dt.date.fromisoformat(m.group(1));return "days_from_date",f"It has been {(now.date()-d).days} day(s) since {d.isoformat()}."
        m=re.match(r"^next weekday\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$",t)
        if m:
            target=["monday","tuesday","wednesday","thursday","friday","saturday","sunday"].index(m.group(1))
            days=(target-now.weekday())%7 or 7;d=now.date()+dt.timedelta(days=days)
            return "next_weekday",f"The next {m.group(1)} is {d.isoformat()}."

        # Number tools.
        m=re.match(r"^(?:square|squared)\s+(-?\d+(?:\.\d+)?)$",t)
        if m:return "square",f"{m.group(1)} squared is {self.fmt(float(m.group(1))**2)}."
        m=re.match(r"^(?:cube|cubed)\s+(-?\d+(?:\.\d+)?)$",t)
        if m:return "cube",f"{m.group(1)} cubed is {self.fmt(float(m.group(1))**3)}."
        m=re.match(r"^(?:square root of|sqrt)\s+(\d+(?:\.\d+)?)$",t)
        if m:return "square_root",f"The square root is {self.fmt(math.sqrt(float(m.group(1))))}."
        m=re.match(r"^(?:what is )?(-?\d+(?:\.\d+)?)\s+(?:to the power of|\^)\s+(-?\d+(?:\.\d+)?)$",t)
        if m:return "power",f"The result is {self.fmt(SafeMath.evaluate(m.group(1)+' ** '+m.group(2)))}."
        m=re.match(r"^(?:what is )?(-?\d+(?:\.\d+)?)\s*%\s+of\s+(-?\d+(?:\.\d+)?)$",t)
        if m:return "percentage",f"{m.group(1)} percent of {m.group(2)} is {self.fmt(float(m.group(1))*float(m.group(2))/100)}."
        m=re.match(r"^percentage change from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)$",t)
        if m:
            a,b=map(float,m.groups())
            return "percentage_change",("Undefined when the starting value is zero." if a==0 else f"Percentage change is {self.fmt((b-a)/abs(a)*100)} percent.")
        m=re.match(r"^(sum|average|mean|median|minimum|maximum|min|max) of\s+(.+)$",t)
        if m:
            vals=self.numbers(m.group(2))
            if not vals:return None
            op=m.group(1);fn={"sum":sum,"average":statistics.mean,"mean":statistics.mean,"median":statistics.median,"minimum":min,"min":min,"maximum":max,"max":max}[op]
            return op,f"The {op} is {self.fmt(fn(vals))}."
        m=re.match(r"^factorial of\s+(\d+)$",t)
        if m:
            n=int(m.group(1))
            if n>500:return "factorial","I can calculate factorials up to 500."
            return "factorial",f"{n}! is {math.factorial(n)}."
        m=re.match(r"^is\s+(\d+)\s+(?:a )?prime$",t)
        if m:
            n=int(m.group(1));return "prime_check",f"{n} is {'prime' if is_prime(n) else 'not prime'}."
        m=re.match(r"^(?:list )?primes up to\s+(\d+)$",t)
        if m:
            n=int(m.group(1))
            if n>10000:return "prime_list","The limit is 10,000 to keep it responsive on the Pi."
            vals=[str(i) for i in range(2,n+1) if is_prime(i)]
            return "prime_list","Primes: "+(", ".join(vals) if vals else "none")
        m=re.match(r"^fibonacci(?: sequence)?(?: up to| for| of)?\s+(\d+)$",t)
        if m:
            n=int(m.group(1))
            if n>50:return "fibonacci","I can list up to 50 Fibonacci terms."
            vals=[];a,b=0,1
            for _ in range(n):vals.append(a);a,b=b,a+b
            return "fibonacci","Fibonacci sequence: "+", ".join(map(str,vals))
        m=re.match(r"^(?:gcd|greatest common divisor) of\s+(-?\d+)\s+(?:and|,)\s+(-?\d+)$",t)
        if m:return "greatest_common_divisor",f"The GCD is {math.gcd(int(m.group(1)),int(m.group(2)))}."
        m=re.match(r"^(?:lcm|least common multiple) of\s+(-?\d+)\s+(?:and|,)\s+(-?\d+)$",t)
        if m:
            a,b=map(int,m.groups());return "least_common_multiple",f"The LCM is {abs(a*b)//math.gcd(a,b) if a and b else 0}."
        m=re.match(r"^round\s+(-?\d+(?:\.\d+)?)\s+to\s+(\d+)\s+decimals?$",t)
        if m:return "round_number",f"Rounded result: {round(float(m.group(1)),min(12,int(m.group(2))))}."
        m=re.match(r"^(?:absolute value of|abs)\s+(-?\d+(?:\.\d+)?)$",t)
        if m:return "absolute_value",f"The absolute value is {abs(float(m.group(1)))}."
        m=re.match(r"^floor\s+(-?\d+(?:\.\d+)?)$",t)
        if m:return "floor_number",f"Floor: {math.floor(float(m.group(1)))}."
        m=re.match(r"^ceil(?:ing)?\s+(-?\d+(?:\.\d+)?)$",t)
        if m:return "ceiling_number",f"Ceiling: {math.ceil(float(m.group(1)))}."
        m=re.match(r"^fraction\s+(-?\d+)\s*/\s*(\d+)\s+as decimal$",t)
        if m:
            d=int(m.group(2))
            return "fraction_to_decimal",("The denominator cannot be zero." if d==0 else f"The decimal is {self.fmt(int(m.group(1))/d)}.")

        # Conversion: Celsius/Fahrenheit/Kelvin and length, mass, volume, speed, time, data.
        m=re.match(r"^(?:convert )?(-?\d+(?:\.\d+)?)\s*°?\s*(c|celsius|f|fahrenheit|k|kelvin)\s+(?:to|in)\s+(c|celsius|f|fahrenheit|k|kelvin)$",t)
        if m:
            v=float(m.group(1));a,b=unit_key(m.group(2)),unit_key(m.group(3))
            c=v if a=="c" else (v-32)*5/9 if a=="f" else v-273.15
            out=c if b=="c" else c*9/5+32 if b=="f" else c+273.15
            return "temperature_conversion",f"{self.fmt(v)} {a.upper()} = {self.fmt(out)} {b.upper()}."
        m=re.match(r"^(?:convert )?(-?\d+(?:\.\d+)?)\s+([a-zA-Z]+)\s+(?:to|in)\s+([a-zA-Z]+)$",original,re.I)
        if m:
            v=float(m.group(1));a,b=unit_key(m.group(2)),unit_key(m.group(3))
            groups={
              "length":{"m":1,"km":1000,"cm":.01,"mm":.001,"mile":1609.344,"miles":1609.344,"mi":1609.344,"ft":.3048,"feet":.3048,"in":.0254,"inch":.0254,"inches":.0254,"yd":.9144,"yard":.9144},
              "mass":{"kg":1,"g":.001,"mg":.000001,"lb":.45359237,"lbs":.45359237,"oz":.028349523125,"ounce":.028349523125,"tonne":1000},
              "volume":{"l":1,"liter":1,"litre":1,"ml":.001,"gallon":3.785411784,"gal":3.785411784,"cup":.2365882365,"cups":.2365882365,"tsp":.00492892159,"tbsp":.0147867648},
              "speed":{"mps":1,"kmh":1/3.6,"kph":1/3.6,"mph":.44704,"fps":.3048},
              "time":{"s":1,"sec":1,"second":1,"seconds":1,"min":60,"minute":60,"minutes":60,"h":3600,"hr":3600,"hour":3600,"hours":3600,"day":86400,"days":86400},
              "data":{"b":1,"byte":1,"bytes":1,"kb":1e3,"mb":1e6,"gb":1e9,"tb":1e12,"kib":1024,"mib":1024**2,"gib":1024**3,"tib":1024**4}}
            family=next((name for name,units in groups.items() if a in units and b in units),None)
            if family:return family+"_conversion",f"{self.fmt(v)} {a} = {self.fmt(v*groups[family][a]/groups[family][b])} {b}."

        # Text processing, only on explicit commands.
        m=re.match(r"^(?:count words in|word count)\s*[:,-]?\s*(.+)$",original,re.I)
        if m:
            words=re.findall(r"\b[\w'-]+\b",m.group(1))
            return "word_count",f"There are {len(words)} word(s)."
        m=re.match(r"^(?:count characters in|character count|count chars)\s*[:,-]?\s*(.+)$",original,re.I)
        if m:return "character_count",f"There are {len(m.group(1))} character(s), including spaces."
        m=re.match(r"^(?:count lines in|line count)\s*[:,-]?\s*(.+)$",original,re.I|re.S)
        if m:return "line_count",f"There are {len(m.group(1).splitlines())} line(s)."
        ops=[
          (r"^(?:uppercase|upper case|make uppercase)\s*[:,-]?\s*(.+)$",str.upper,"uppercase"),
          (r"^(?:lowercase|lower case|make lowercase)\s*[:,-]?\s*(.+)$",str.lower,"lowercase"),
          (r"^(?:title case|titlecase)\s*[:,-]?\s*(.+)$",str.title,"title_case"),
          (r"^(?:reverse text|reverse)\s*[:,-]?\s*(.+)$",lambda s:s[::-1],"reverse_text"),
          (r"^(?:sort words|alphabetize words)\s*[:,-]?\s*(.+)$",lambda s:" ".join(sorted(s.split(),key=str.casefold)),"sort_words"),
          (r"^(?:unique words|remove duplicate words)\s*[:,-]?\s*(.+)$",unique_words,"unique_words"),
          (r"^(?:collapse spaces|clean spaces)\s*[:,-]?\s*(.+)$",lambda s:" ".join(s.split()),"collapse_spaces"),
          (r"^remove punctuation\s*[:,-]?\s*(.+)$",lambda s:re.sub(r"[^\w\s]","",s),"remove_punctuation"),
          (r"^slugify\s*[:,-]?\s*(.+)$",slugify,"slugify"),
          (r"^(?:get initials|initials of)\s*[:,-]?\s*(.+)$",lambda s:"".join(w[0].upper() for w in s.split() if w),"initials"),
          (r"^(?:make acronym|acronym of)\s*[:,-]?\s*(.+)$",lambda s:"".join(w[0].upper() for w in s.split() if w),"acronym"),
          (r"^(?:trim text|trim)\s*[:,-]?\s*(.+)$",str.strip,"text_trim")]
        for pattern,fn,route in ops:
            m=re.match(pattern,original,re.I|re.S)
            if m:return route,str(fn(m.group(1)))[:2500]
        m=re.match(r"^(?:is this a palindrome|palindrome check)\s*[:,-]?\s*(.+)$",original,re.I)
        if m:
            s=re.sub(r"[\W_]+","",m.group(1).casefold(),flags=re.UNICODE)
            return "palindrome_check","Yes, that is a palindrome." if s==s[::-1] else "No, that is not a palindrome."
        m=re.match(r"^(?:word frequency|frequency of words)\s*[:,-]?\s*(.+)$",original,re.I)
        if m:
            words=re.findall(r"\b\w+\b",m.group(1).casefold(),re.UNICODE)
            counts=sorted(((w,words.count(w)) for w in set(words)),key=lambda x:(-x[1],x[0]))[:12]
            return "word_frequency","Word frequency: "+", ".join(f"{w}: {n}" for w,n in counts)
        m=re.match(r"^(?:count vowels in|vowel count)\s*[:,-]?\s*(.+)$",original,re.I)
        if m:return "vowel_count",f"Vowel count: {sum(c.casefold() in 'aeiou' for c in m.group(1))}."
        m=re.match(r"^(?:count consonants in|consonant count)\s*[:,-]?\s*(.+)$",original,re.I)
        if m:return "consonant_count",f"Consonant count: {sum(c.isalpha() and c.casefold() not in 'aeiou' for c in m.group(1))}."
        m=re.match(r"^repeat text\s+(\d+)\s+times?\s*[:,-]?\s*(.+)$",original,re.I)
        if m:
            n=int(m.group(1))
            return "repeat_text",("I can repeat text up to 30 times." if n>30 else "\n".join([m.group(2)]*n)[:2500])
        m=re.match(r"^(?:character codes|unicode codes)\s*[:,-]?\s*(.{1,80})$",original,re.I)
        if m:return "character_codes","Character codes: "+", ".join(f"{c!r}={ord(c)}" for c in m.group(1)[:30])
        m=re.match(r"^replace\s+(.+?)\s+with\s+(.+?)\s+in\s+(.+)$",original,re.I)
        if m:
            old,new,value=m.groups();return "replace_text",value.replace(old,new)[:2500]

        # Encoders and JSON formatters.
        encoders=[
          (r"^base64 encode\s*[:,-]?\s*(.+)$","base64_encode"),
          (r"^url encode\s*[:,-]?\s*(.+)$","url_encode"),
          (r"^binary encode\s*[:,-]?\s*(.+)$","binary_encode"),
          (r"^hex encode\s*[:,-]?\s*(.+)$","hex_encode")]
        for pattern,route in encoders:
            m=re.match(pattern,original,re.I)
            if m:
                s=m.group(1)
                if route=="base64_encode":out=base64.b64encode(s.encode()).decode()
                elif route=="url_encode":out=urllib.parse.quote(s,safe="")
                elif route=="binary_encode":out=" ".join(format(b,"08b") for b in s.encode())
                else:out=s.encode().hex()
                return route,out
        m=re.match(r"^base64 decode\s*[:,-]?\s*(.+)$",original,re.I)
        if m:
            try:return "base64_decode",base64.b64decode(m.group(1),validate=True).decode("utf-8")
            except Exception:return "base64_decode","That isn't valid UTF-8 base64 text."
        m=re.match(r"^url decode\s*[:,-]?\s*(.+)$",original,re.I)
        if m:return "url_decode",urllib.parse.unquote(m.group(1))
        m=re.match(r"^binary decode\s*[:,-]?\s*(.+)$",original,re.I)
        if m:
            try:
                raw=re.sub(r"\s+","",m.group(1));groups=re.findall(r"[01]{8}",raw)
                if not groups or "".join(groups)!=raw:raise ValueError
                return "binary_decode",bytes(int(x,2) for x in groups).decode("utf-8")
            except Exception:return "binary_decode","Use eight-bit binary bytes separated by spaces."
        m=re.match(r"^hex decode\s*[:,-]?\s*(.+)$",original,re.I)
        if m:
            try:return "hex_decode",bytes.fromhex(m.group(1)).decode("utf-8")
            except Exception:return "hex_decode","That isn't valid UTF-8 hexadecimal text."
        for pattern,indent,route in ((r"^json pretty\s*[:,-]?\s*(.+)$",2,"json_pretty"),(r"^json compact\s*[:,-]?\s*(.+)$",None,"json_compact")):
            m=re.match(pattern,original,re.I|re.S)
            if m:
                try:
                    obj=json.loads(m.group(1));return route,json.dumps(obj,ensure_ascii=False,indent=indent,separators=None if indent else (",",":"))[:2500]
                except json.JSONDecodeError as exc:return route,f"Invalid JSON near character {exc.pos}."
        m=re.match(r"^number to binary\s+(-?\d+)$",t)
        if m:return "binary_encode",f"Binary: {bin(int(m.group(1)))}"
        m=re.match(r"^binary to number\s+([01]+)$",t)
        if m:return "binary_decode",f"Decimal: {int(m.group(1),2)}"
        m=re.match(r"^number to hex\s+(-?\d+)$",t)
        if m:return "hex_encode",f"Hexadecimal: {hex(int(m.group(1)))}"
        m=re.match(r"^hex to number\s+([0-9a-f]+)$",t)
        if m:return "hex_decode",f"Decimal: {int(m.group(1),16)}"

        # Random and offline fun.
        m=re.match(r"^(?:random number|pick a number between)\s+(-?\d+)\s+(?:and|to)\s+(-?\d+)$",t)
        if m:
            a,b=sorted((int(m.group(1)),int(m.group(2))))
            if b-a>10**12:return "random_number","Please choose a smaller range."
            return "random_number",f"You got {random.randint(a,b)}."
        if t in {"flip a coin","coin flip","toss a coin"}:return "coin_flip",random.choice(["Heads!","Tails!"])
        if t in {"roll a die","roll a dice","roll dice"}:return "dice_roll",f"You rolled a {random.randint(1,6)}."
        m=re.match(r"^(?:choose from|pick one of)\s+(.+)$",original,re.I)
        if m:
            choices=[x.strip() for x in re.split(r",| or ",m.group(1)) if x.strip()]
            if choices:return "random_choice","I choose "+random.choice(choices)+"."
        if t in {"generate a password","make a password","generate password"}:
            alphabet=string.ascii_letters+string.digits+"!@#$%&*+-_"
            return "password_generator","Generated password: "+"".join(random.SystemRandom().choice(alphabet) for _ in range(16))
        if t in {"generate uuid","make a uuid","new uuid"}:return "uuid_generator",str(uuid.uuid4())
        if t in {"tell me a joke","make me laugh","joke"}:
            return "joke",random.choice(["Why did the computer get cold? It left its Windows open.","Why did the robot go on holiday? It needed to recharge.","What do you call a sleeping laptop? A nap-top.","Why was the math book worried? It had too many problems."])
        if t in {"tell me a riddle","riddle"}:return "riddle","Riddle: What has keys but can't open locks? Answer: a piano."
        if t in {"tell me a fun fact","trivia","give me trivia"}:
            return "trivia",random.choice(["Octopuses have three hearts.","A day on Venus is longer than its year.","Honeybees communicate food locations with a waggle dance.","The Eiffel Tower can grow slightly taller in hot weather.","Bananas are botanically berries; strawberries are not."])
        if t in {"say something nice","compliment me","give me a compliment"}:
            return "compliment",random.choice(["You are learning by building something real. Keep going!","Curiosity is a superpower. Keep asking good questions.","Small, steady improvements add up to impressive projects."])

        expr=re.sub(r"^(?:calculate|compute|solve|what is|what's)\s+","",original,flags=re.I)
        if re.fullmatch(r"[0-9\s.+\-*/()%^]+",expr) and re.search(r"\d",expr):
            try:return "calculator",f"The answer is {self.fmt(SafeMath.evaluate(expr))}."
            except ZeroDivisionError:return "calculator","Division by zero is undefined."
            except (ValueError,SyntaxError,OverflowError,TypeError):return None
        return None

    def _start_timer(self,seconds,label):
        tid=uuid.uuid4().hex[:8]
        def fire():
            with self.lock:self.timers.pop(tid,None)
            if self.notify:self.notify("Timer finished: "+label+".")
        timer=threading.Timer(seconds,fire);timer.daemon=True
        with self.lock:self.timers[tid]=timer
        timer.start()
        return tid
    @staticmethod
    def duration(seconds):
        total=max(0,int(seconds));days,rem=divmod(total,86400);hours,rem=divmod(rem,3600);minutes,secs=divmod(rem,60)
        parts=[]
        if days:parts.append(f"{days} day(s)")
        if hours:parts.append(f"{hours} hour(s)")
        if minutes:parts.append(f"{minutes} minute(s)")
        if not parts:parts.append(f"{secs} second(s)")
        return ", ".join(parts)
    def _memory(self):
        try:
            values={}
            for line in Path("/proc/meminfo").read_text().splitlines():
                k,v=line.split(":",1);values[k]=int(v.strip().split()[0])*1024
            total,avail=values["MemTotal"],values.get("MemAvailable",values.get("MemFree",0))
            return f"RAM: {avail/1024**2:.0f} MiB available of {total/1024**2:.0f} MiB."
        except Exception:return "Memory details are unavailable on this operating system."
    def _cpu_temp(self):
        for path in (Path("/sys/class/thermal/thermal_zone0/temp"),Path("/sys/devices/virtual/thermal/thermal_zone0/temp")):
            try:
                v=float(path.read_text().strip());return v/1000 if v>200 else v
            except Exception:pass
        return None
    @staticmethod
    def _local_ip():
        sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        try:sock.connect(("192.0.2.1",80));return sock.getsockname()[0]
        except Exception:return "unavailable"
        finally:sock.close()
    def close(self):
        with self.lock:timers=list(self.timers.values());self.timers.clear()
        for timer in timers:timer.cancel()
        try:self.db.close()
        except Exception:pass
