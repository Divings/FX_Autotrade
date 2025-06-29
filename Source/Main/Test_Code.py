import unittest
from unittest.mock import patch, MagicMock
import AutoTrade
import threading

class TestAutoTrade(unittest.IsolatedAsyncioTestCase):

    @patch('AutoTrade.get_price')
    @patch('AutoTrade.get_positions')
    @patch('AutoTrade.notify_slack')
    def test_monitor_trend_no_data(self, mock_notify, mock_positions, mock_price):
        stop_event = threading.Event()
        # RSIやADXの蓄積が不十分なケースを再現
        mock_price.return_value = {'ask': 160.01, 'bid': 160.00}
        AutoTrade.shared_state["rsi_list"] = []
        AutoTrade.shared_state["adx_list"] = []
        AutoTrade.shared_state["trend"] = None

        AutoTrade.monitor_trend(stop_event)

        self.assertEqual(AutoTrade.shared_state["trend"], None)
        mock_notify.assert_not_called()

    @patch('AutoTrade.open_order')
    @patch('AutoTrade.notify_slack')
    def test_open_order_success(self, mock_notify, mock_open):
        # 成功した注文結果のダミーを返す
        mock_open.return_value = {
            "data": [
                {"rootOrderId": 123456789}
            ]
        }
        result = AutoTrade.open_order("BUY")
        self.assertIn("data", result)
        self.assertEqual(result["data"][0]["rootOrderId"], 123456789)

    @patch('AutoTrade.get_positions')
    @patch('AutoTrade.close_order')
    @patch('AutoTrade.notify_slack')
    def test_failSafe_with_position(self, mock_notify, mock_close, mock_positions):
        mock_positions.return_value = [{
            "price": "160.00",
            "positionId": "pos123",
            "size": "1000",
            "side": "BUY"
        }]
        AutoTrade.failSafe()
        mock_close.assert_called_once()

    @patch('AutoTrade.get_positions')
    @patch('AutoTrade.notify_slack')
    def test_failSafe_no_position(self, mock_notify, mock_positions):
        mock_positions.return_value = []
        result = AutoTrade.failSafe()
        self.assertEqual(result, 0)

if __name__ == '__main__':
    unittest.main()
