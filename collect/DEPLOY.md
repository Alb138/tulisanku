# Hosting the collection page on PythonAnywhere (free)

Gives a permanent link, `https://<username>.pythonanywhere.com`, that works without your laptop.
Drawings are stored on PythonAnywhere (512 MB free; one person's 62 drawings take ~1-2 MB).

## One-time setup (about 15 minutes)

1. **Create a free "Beginner" account** at https://www.pythonanywhere.com.
   Your username becomes part of the link.

2. **Make a secret export key** on your laptop and keep it private (e.g. in a password manager).
   It is the password for downloading the drawings later.
   ```
   .venv\Scripts\python -c "import secrets; print(secrets.token_urlsafe(24))"
   ```

3. On PythonAnywhere open **Consoles → Bash** and run:
   ```
   git clone https://github.com/Alb138/tulisanku.git
   ```

4. Open the **Web** tab → **Add a new web app** → *Next* → **Manual configuration**
   → choose the newest Python version → *Next*.

5. On the Web tab, click the **WSGI configuration file** link
   (`/var/www/<username>_pythonanywhere_com_wsgi.py`). Delete everything in it, **type** the
   lines below (fill in your key), and **Save**:
   ```python
   import os, sys
   sys.path.insert(0, os.path.expanduser("~/tulisanku"))
   os.environ["TULISANKU_EXPORT_KEY"] = "<your secret key>"
   from collect.app import app as application
   ```
   This file lives only on PythonAnywhere, so the key never reaches GitHub.

6. Back on the Web tab, click **Reload**, then open `https://<username>.pythonanywhere.com`
   (press Ctrl+F5 to skip the browser's cached copy). You should see the TULISANKU consent
   page. Share that link with your friends.

## If it shows "Something went wrong"

Open the **error log** link on the Web tab and read the **newest entries at the bottom**.
Times are in UTC (Jakarta time minus 7 hours) - make sure you are reading an error from
after your last change. Changes only take effect after **Save** and then **Reload**.

| Error in the log | Cause and fix |
|---|---|
| `IndentationError: unexpected indent` / `SyntaxError` on line 1 | A space before a line, or an invisible character copied from a chat or web page. Delete the file's content and *type* the 4 lines. |
| `ModuleNotFoundError: No module named 'collect'` | Python can't find the code: run `ls ~/tulisanku` in a Bash console; if it is missing, `git clone` it (step 3). Use `os.path.expanduser("~/tulisanku")` so upper/lower case in the username can't matter. |
| An error mentioning Flask | Flask is too old: `pip3 install --user --upgrade flask` (same Python version as the web app), then Reload. |

## Every month

Free web apps stop after one month: on the Web tab click **Run until 1 month from today**.

## After changing the code

In the Bash console: `cd ~/tulisanku && git pull`, then click **Reload** on the Web tab.

## Getting the drawings onto your laptop

```
.venv\Scripts\python scripts\fetch_collected.py --url https://<username>.pythonanywhere.com
```

It asks for the export key and adds any new sessions to `data/raw/handwriting/drawn/`.
