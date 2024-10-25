# production.py
import multiprocessing
import gunicorn.app.base
from main import app

class StandaloneApplication(gunicorn.app.base.BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items()
                 if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

if __name__ == '__main__':
    options = {
        'bind': '%s:%s' % ('0.0.0.0', '8000'),
        'workers': multiprocessing.cpu_count() * 2 + 1,
        'worker_class': 'uvicorn.workers.UvicornWorker',
        'timeout': 120,
        'keepalive': 5,
        'errorlog': '-',
        'accesslog': '-',
        'access_log_format': '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
    }

    StandaloneApplication(app, options).run()