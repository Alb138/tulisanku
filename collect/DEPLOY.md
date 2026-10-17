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
   (`/var/www/<username>_pythonanywhere_com_wsgi.py`). Delete everything in it, paste the
   lines below (fill in your username and key), and **Save**:
   ```python
   import os, sys
   sys.path.insert(0, "/home/<username>/tulisanku")
   os.environ["TULISANKU_EXPORT_KEY"] = "<your secret key>"
   from collect.app import app as application
   ```
   This file lives only on PythonAnywhere, so the key never reaches GitHub.

6. Back on the Web tab, click **Reload**, then open `https://<username>.pythonanywhere.com`.
   You should see the TULISANKU consent page. Share that link with your friends.

If the page shows an error, open the **error log** link on the Web tab. An error mentioning
Flask usually means its version is too old: in the Bash console run
`pip3 install --user --upgrade flask` (use the same Python version as the web app), then Reload.

## Every month

Free web apps stop after one month: on the Web tab click **Run until 1 month from today**.

## After changing the code

In the Bash console: `cd ~/tulisanku && git pull`, then click **Reload** on the Web tab.

## Getting the drawings onto your laptop

```
.venv\Scripts\python scripts\fetch_collected.py --url https://<username>.pythonanywhere.com
```

It asks for the export key and adds any new sessions to `data/raw/handwriting/drawn/`.
