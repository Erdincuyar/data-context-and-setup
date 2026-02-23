import pandas as pd
import numpy as np
from olist.data import Olist
from olist.order import Order


class Seller:
    def __init__(self):
        # Verileri bir kez yükle
        olist = Olist()
        self.data = olist.get_data()
        self.order = Order()

    def get_seller_features(self):
        """
        Geriye şunu döndürür: 'seller_id', 'seller_city', 'seller_state'
        """
        sellers = self.data['sellers'].copy()
        sellers.drop('seller_zip_code_prefix', axis=1, inplace=True)
        sellers.drop_duplicates(inplace=True)
        return sellers

    def get_seller_delay_wait_time(self):
        """
        Geriye şunu döndürür: 'seller_id', 'delay_to_carrier', 'wait_time'
        """
        order_items = self.data['order_items'].copy()
        orders = self.data['orders'].query("order_status=='delivered'").copy()

        ship = order_items.merge(orders, on='order_id')

        # Tarih formatlarını düzenle
        ship.loc[:, 'shipping_limit_date'] = pd.to_datetime(ship['shipping_limit_date'])
        ship.loc[:, 'order_delivered_carrier_date'] = pd.to_datetime(ship['order_delivered_carrier_date'])
        ship.loc[:, 'order_delivered_customer_date'] = pd.to_datetime(ship['order_delivered_customer_date'])
        ship.loc[:, 'order_purchase_timestamp'] = pd.to_datetime(ship['order_purchase_timestamp'])

        # Gecikme ve bekleme süresi hesaplama
        def delay_to_logistic_partner(d):
            days = np.mean((d.order_delivered_carrier_date - d.shipping_limit_date) / np.timedelta64(24, 'h'))
            return max(0, days)

        def order_wait_time(d):
            days = np.mean((d.order_delivered_customer_date - d.order_purchase_timestamp) / np.timedelta64(24, 'h'))
            return days

        delay = ship.groupby('seller_id').apply(delay_to_logistic_partner).reset_index()
        delay.columns = ['seller_id', 'delay_to_carrier']

        wait = ship.groupby('seller_id').apply(order_wait_time).reset_index()
        wait.columns = ['seller_id', 'wait_time']

        return delay.merge(wait, on='seller_id')

    def get_active_dates(self):
        """
        Geriye şunu döndürür: 'seller_id', 'date_first_sale', 'date_last_sale', 'months_on_olist'
        """
        orders_approved = self.data['orders'][['order_id', 'order_approved_at']].dropna()
        orders_sellers = orders_approved.merge(self.data['order_items'], on='order_id')[['order_id', 'seller_id', 'order_approved_at']].drop_duplicates()
        orders_sellers["order_approved_at"] = pd.to_datetime(orders_sellers["order_approved_at"])

        df = orders_sellers.groupby('seller_id').agg({
            "order_approved_at": ["min", "max"]
        })
        df.columns = ["date_first_sale", "date_last_sale"]
        
        # Ay sayısını hesapla (0 ayı önlemek için 1 ekleyebilirsin veya max(1, ...) yapabilirsin)
        df['months_on_olist'] = np.ceil((df['date_last_sale'] - df['date_first_sale']) / np.timedelta64(30, 'D'))
        return df.reset_index()

    def get_quantity(self):
        """
        Geriye şunu döndürür: 'seller_id', 'n_orders', 'quantity', 'quantity_per_order'
        """
        order_items = self.data['order_items']
        n_orders = order_items.groupby('seller_id')['order_id'].nunique().reset_index(name='n_orders')
        quantity = order_items.groupby('seller_id')['order_id'].count().reset_index(name='quantity')

        result = n_orders.merge(quantity, on='seller_id')
        result['quantity_per_order'] = result['quantity'] / result['n_orders']
        return result

    def get_sales(self):
        """
        Geriye şunu döndürür: 'seller_id', 'sales'
        """
        return self.data['order_items'][['seller_id', 'price']]\
            .groupby('seller_id')\
            .sum()\
            .rename(columns={'price': 'sales'})

    def get_review_score(self):
        """
        Geriye şunu döndürür: 'seller_id', 'share_of_five_stars', 'share_of_one_stars', 
        'review_score', 'cost_of_reviews'
        """
        orders_reviews = self.data['order_reviews'].copy()
        orders_items = self.data['order_items'].copy()

        matching_table = orders_items[['order_id', 'seller_id']].drop_duplicates()
        df = matching_table.merge(orders_reviews[['order_id', 'review_score']], on='order_id')

        # 💰 Review Maliyet Tablosu
        costs = {1: 100, 2: 50, 3: 40, 4: 0, 5: 0}
        df['dim_cost_review'] = df['review_score'].map(costs)
        
        df['is_five_star'] = df['review_score'].map(lambda x: 1 if x == 5 else 0)
        df['is_one_star'] = df['review_score'].map(lambda x: 1 if x == 1 else 0)

        result = df.groupby('seller_id').agg({
            'is_five_star': 'mean',
            'is_one_star': 'mean',
            'review_score': 'mean',
            'dim_cost_review': 'sum'
        }).reset_index()

        result.columns = ['seller_id', 'share_of_five_stars', 'share_of_one_stars', 'review_score', 'cost_of_reviews']
        return result

    def get_training_data(self):
        """
        Ana birleştirme metodu: revenues, cost_of_reviews ve profits sütunlarını içerir.
        """
        # Tüm metodları merge et
        training_set = self.get_seller_features()\
            .merge(self.get_seller_delay_wait_time(), on='seller_id')\
            .merge(self.get_active_dates(), on='seller_id')\
            .merge(self.get_quantity(), on='seller_id')\
            .merge(self.get_sales(), on='seller_id')\
            .merge(self.get_review_score(), on='seller_id')

        # 💸 Revenues: Abonelik (80/ay) + Komisyon (%10 satış)
        training_set['revenues'] = (training_set['months_on_olist'] * 80) + (training_set['sales'] * 0.1)

        # 📉 Profits: Gelir - Kötü yorum maliyeti
        training_set['profits'] = training_set['revenues'] - training_set['cost_of_reviews']

        return training_set