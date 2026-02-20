import json
from pathlib import Path
import tempfile
import time
import unittest

from app.services.join_invites import JoinInviteService


class JoinInviteTests(unittest.TestCase):
    def test_create_and_consume(self):
        service = JoinInviteService(default_ttl_s=60)
        invite = service.create(issued_by="test")
        self.assertTrue(service.is_valid(invite.ticket_id))
        consumed = service.consume(invite.ticket_id)
        self.assertTrue(consumed.used)
        self.assertFalse(service.is_valid(invite.ticket_id))

    def test_expire(self):
        service = JoinInviteService(default_ttl_s=1)
        invite = service.create(issued_by="test", ttl_s=1)
        time.sleep(1.05)
        self.assertFalse(service.is_valid(invite.ticket_id))
        with self.assertRaises(ValueError):
            service.consume(invite.ticket_id)

    def test_invite_persists_across_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "join_invites.json"
            service = JoinInviteService(default_ttl_s=60, state_path=path)
            invite = service.create(issued_by="test", ttl_s=60)

            restarted = JoinInviteService(default_ttl_s=60, state_path=path)
            loaded = restarted.get(invite.ticket_id)
            self.assertIsNotNone(loaded)
            self.assertTrue(restarted.is_valid(invite.ticket_id))
            self.assertAlmostEqual(float(loaded.expires_at), float(invite.expires_at), places=3)

    def test_consumed_invite_stays_consumed_after_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "join_invites.json"
            service = JoinInviteService(default_ttl_s=60, state_path=path)
            invite = service.create(issued_by="test", ttl_s=60)
            service.consume(invite.ticket_id)

            restarted = JoinInviteService(default_ttl_s=60, state_path=path)
            loaded = restarted.get(invite.ticket_id)
            self.assertIsNotNone(loaded)
            self.assertTrue(bool(loaded.used))
            self.assertEqual(int(loaded.uses), 1)
            self.assertFalse(restarted.is_valid(invite.ticket_id))

    def test_expired_invites_are_pruned_on_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "join_invites.json"
            service = JoinInviteService(default_ttl_s=60, state_path=path)
            invite = service.create(issued_by="test", ttl_s=60)

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["invites"][0]["expires_at"] = time.time() - 5.0
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            restarted = JoinInviteService(default_ttl_s=60, state_path=path)
            self.assertIsNone(restarted.get(invite.ticket_id))
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved.get("invites"), [])


if __name__ == "__main__":
    unittest.main()
