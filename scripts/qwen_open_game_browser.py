#!/usr/bin/env python3
"""Actual browser smoke for Qwen-authored, Kairos-applied, Studio-produced game."""
from pathlib import Path
import hashlib
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
import json
import threading
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"open-agent-evidence"
WORK=ROOT/"studio_workspace"
GAME=WORK/"output/index.html"
class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,directory=str(WORK),**kwargs)
    def log_message(self,*args):pass
def main():
    assert GAME.is_file(),"No Studio-produced game to test"
    js_errors=[]
    checks=[]
    http=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    t=threading.Thread(target=http.serve_forever,daemon=True);t.start()
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,args=["--no-sandbox"])
            page=browser.new_page(viewport={"width":390,"height":844},device_scale_factor=1)
            page.on("pageerror",lambda error:js_errors.append(str(error)))
            page.goto("http://127.0.0.1:"+str(http.server_port)+"/output/index.html")
            assert page.locator("#scene").evaluate("(el)=>!!el.getContext('2d')")
            assert page.locator("#garage .slot").count()==3
            checks.append("Canvas and three-slot garage rendered")
            page.screenshot(path=str(OUT/"game-before-battle.png"),full_page=True)
            page.locator("#start").click()
            page.wait_for_timeout(250)
            assert page.evaluate("fight !== null && fight.enemies.length >= 3")
            checks.append("Combat starts and enemies spawn")
            page.locator("#fire").click()
            assert page.evaluate("fight.shots.length >= 1")
            checks.append("Firing produces projectile")
            page.locator("#pause-btn").click()
            assert page.evaluate("paused === true")
            page.locator("#pause-btn").click()
            assert page.evaluate("paused === false")
            checks.append("Previously implemented Pause/Resume still works")
            page.screenshot(path=str(OUT/"game-in-battle.png"),full_page=True)
            assert not js_errors, "Browser errors: "+str(js_errors)
            checks.append("No uncaught JavaScript errors during smoke")
            browser.close()
    finally:http.shutdown()
    result={"status":"PASS","gate":"G4 real browser smoke of model-written game",
       "checks":checks,"js_errors":js_errors,
       "game_sha256":hashlib.sha256(GAME.read_bytes()).hexdigest(),
       "product_accepted":False,
       "unverified":["visual fidelity to customer reference","full gameplay quality",
         "all intended features","production deployment","customer approval"]}
    (OUT/"gate4-browser.json").write_text(json.dumps(result,indent=2)+"\n")
    print("OPEN_QWEN_BROWSER_PASS",json.dumps(result),flush=True)
if __name__=="__main__":main()
