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


if __name__ == "__main__":
    unittest.main()
