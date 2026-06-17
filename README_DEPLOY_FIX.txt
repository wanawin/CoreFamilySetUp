DEPLOY FIX

The Streamlit error shown is: Main module does not exist.
That means Streamlit Cloud is looking for an app file name that is not present in the GitHub repo.

This package includes the same v10 app under multiple safe entrypoint names:
- core_affinity_lab_v1.py       <-- use this if Streamlit Cloud main file path is core_affinity_lab_v1.py
- streamlit_app.py              <-- Streamlit Cloud default
- app.py                        <-- common fallback
- core_affinity_lab_v10_DEPLOY_ALIAS_FIX.py

Fastest fix:
1. Upload/commit ALL files in this package to the root of the GitHub repo.
2. In Streamlit Cloud, set Main file path to: core_affinity_lab_v1.py
3. Reboot/redeploy the app.

No mining/scoring/profile logic changed from the v9 crash-fix app. This package only fixes deploy entrypoint naming.
