# Участие в разработке

1. Создайте fork и отдельную ветку.
2. Установите только лёгкое окружение: `py -3.10 -m venv .venv`, затем `.\.venv\Scripts\pip install -r requirements.txt -r requirements-dev.txt`.
3. Для UI без моделей задайте `$env:DUBBING_STUDIO_MOCK='1'` и запустите `start.ps1`.
4. Перед pull request выполните `pytest -q`, `python -m compileall -q app workers tools tests` и `node --check static/app.js`.

Не коммитьте модели, виртуальные окружения, пользовательские видео, результаты и логи. Изменения голосового клонирования должны сохранять явное требование согласия владельца голоса.
