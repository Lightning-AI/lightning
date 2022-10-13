import os
from pathlib import Path
from time import sleep
from typing import Optional
from uuid import uuid4

import pytest

from lightning_app import LightningApp, LightningFlow, LightningWork
from lightning_app.components.database import Database, DatabaseClient
from lightning_app.components.database.utilities import GeneralModel
from lightning_app.runners import MultiProcessRuntime
from lightning_app.utilities.imports import _is_sqlmodel_available

if _is_sqlmodel_available():
    from sqlmodel import Field, SQLModel

    class TestConfig(SQLModel, table=True):
        __table_args__ = {"extend_existing": True}

        id: Optional[int] = Field(default=None, primary_key=True)
        name: str


class Work(LightningWork):
    def __init__(self):
        super().__init__(parallel=True)
        self.done = False

    def run(self, client: DatabaseClient):
        rows = client.select_all()
        while len(rows) == 0:
            print(rows)
            sleep(0.1)
            rows = client.select_all()
        self.done = True


@pytest.mark.skipif(not _is_sqlmodel_available(), reason="sqlmodel is required for this test.")
def test_client_server():

    database_path = Path("database.db").resolve()
    if database_path.exists():
        os.remove(database_path)

    general = GeneralModel.from_obj(TestConfig(name="name"), token="a")
    assert general.cls_name == "TestConfig"
    assert general.data == '{"id": null, "name": "name"}'

    class Flow(LightningFlow):
        def __init__(self):
            super().__init__()
            self._token = str(uuid4())
            self.db = Database(models=[TestConfig])
            self._client = None
            self.tracker = None
            self.work = Work()

        def run(self):
            self.db.run(token=self._token)

            if not self.db.alive():
                return

            if not self._client:
                self._client = DatabaseClient(model=TestConfig, db_url=self.db.url, token=self._token)

            assert self._client

            self.work.run(self._client)

            if self.tracker is None:
                self._client.insert(TestConfig(name="name"))
                elem = self._client.select_all(TestConfig)[0]
                assert elem.name == "name"
                self.tracker = "update"

            elif self.tracker == "update":
                elem = self._client.select_all(TestConfig)[0]
                elem.name = "new_name"
                self._client.update(elem)

                elem = self._client.select_all(TestConfig)[0]
                assert elem.name == "new_name"
                self.tracker = "delete"

            elif self.tracker == "delete" and self.work.done:
                self.work.stop()

                elem = self._client.select_all(TestConfig)[0]
                elem = self._client.delete(elem)

                assert not self._client.select_all(TestConfig)
                self._client.insert(TestConfig(name="name"))

                assert self._client.select_all(TestConfig)
                self._exit()

    app = LightningApp(Flow())
    MultiProcessRuntime(app, start_server=False).dispatch()

    database_path = Path("database.db").resolve()
    if database_path.exists():
        os.remove(database_path)


@pytest.mark.skipif(not _is_sqlmodel_available(), reason="sqlmodel is required for this test.")
def test_work_database_restart():

    id = str(uuid4()).split("-")[0]

    class Flow(LightningFlow):
        def __init__(self, restart=False):
            super().__init__()
            self.db = Database(db_filename=id, models=[TestConfig])
            self._client = None
            self.restart = restart

        def run(self):
            self.db.run()

            if not self.db.alive():
                return
            elif not self._client:
                self._client = DatabaseClient(self.db.db_url, None, model=TestConfig)

            if not self.restart:
                self._client.insert(TestConfig(name="echo"))
                self._exit()
            else:
                assert os.path.exists(id)
                assert len(self._client.select_all()) == 1
                self._exit()

    app = LightningApp(Flow())
    MultiProcessRuntime(app).dispatch()

    # Note: Waiting for SIGTERM signal to be handled
    sleep(2)

    os.remove(id)

    app = LightningApp(Flow(restart=True))
    MultiProcessRuntime(app).dispatch()

    # Note: Waiting for SIGTERM signal to be handled
    sleep(2)

    os.remove(id)
