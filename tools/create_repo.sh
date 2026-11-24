#!/usr/bin/env bash
# Create GitHub repo + push + release(assets). mac/Linux, needs gh auth + dist/
# usage: tools/create_repo.sh owner/name [tag]
set -euo pipefail
REPO="${1:?usage: tools/create_repo.sh owner/name [tag]}"
TAG="${2:-v6.2.0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DESC="DDSP-SVC 6.2 + MSST WebUI Windows 开箱即用版：单套 torch 2.7.1+cu128 覆盖 RTX 30/40/50 系，无显卡回落 CPU，fairseq 纯 Python 内置免编译。WebUI 7860 / TensorBoard 6006"
TOPICS=(ddsp ddsp-svc voice-conversion singing-voice-conversion rectified-flow reflow pytorch cuda
  windows portable fairseq hubert contentvec rmvpe hifigan music-source-separation demucs gradio webui audio)

gh repo create "$REPO" --public --description "$DESC" --source . --remote origin --push
for t in "${TOPICS[@]}"; do gh repo edit "$REPO" --add-topic "$t" >/dev/null; done

if [ -d dist ]; then
  NOTES=".github/release-notes/$TAG.md"
  gh release create "$TAG" dist/*.zip.00* \
     --title "$TAG · Windows 开箱即用（cu128 单环境 / 30·40·50 系通用）" \
     ${NOTES:+--notes-file "$NOTES"}
  python3 - "$REPO" "$TAG" <<'PY'
import json,sys,pathlib
repo,tag=sys.argv[1],sys.argv[2]
p=pathlib.Path('tools/release.json'); d=json.loads(p.read_text(encoding='utf-8'))
d['repo'],d['tag']=repo,tag
for a in d['assets']:
    for part in a['parts']:
        part['url']=f"https://github.com/{repo}/releases/download/{tag}/{part['name']}"
p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
print('release.json updated')
PY
  git add tools/release.json && git commit -m "chore(release): 指向 $TAG 附件" && git push
fi
echo "DONE -> https://github.com/$REPO"
