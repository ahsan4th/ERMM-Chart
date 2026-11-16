# Publish the repository and the web app

## 1. Push to GitHub (once)

1. On https://github.com/new create an **empty** public repository named
   `ERMM-Chart` under the account `ahsan4th` (no README, no licence, no .gitignore).
2. In a terminal, inside the unzipped folder `ERMM-Chart`:

```bash
git remote add origin https://github.com/ahsan4th/ERMM-Chart.git
git push -u origin main
```

GitHub asks for the user name and a personal access token
(Settings → Developer settings → Personal access tokens) instead of the password.
With GitHub Desktop: File → Add local repository → choose the folder → Publish repository.

## 2. Deploy the app on Streamlit Community Cloud (free)

1. Sign in at https://share.streamlit.io with the GitHub account.
2. Create app → repository `ahsan4th/ERMM-Chart`, branch `main`, main file `app.py`.
3. Advanced settings → Python 3.12. Deploy.

The app address (for example `https://ermm-chart.streamlit.app`) can be added
to the README and, if desired, to Section 7 and the Code availability statement
of the article.

## 3. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 4. Check the installation

```bash
pip install -r requirements-dev.txt
pytest tests
```
