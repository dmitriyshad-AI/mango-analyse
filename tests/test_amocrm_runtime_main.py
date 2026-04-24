from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from mango_mvp.amocrm_runtime.main import app


class AmoRuntimeMainTest(unittest.TestCase):
    def test_legacy_unprefixed_routes_are_not_exposed(self) -> None:
        client = TestClient(app)

        response = client.get("/integrations/amocrm/status")

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
