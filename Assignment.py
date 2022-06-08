{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "bd470713",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "c5aa5878",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 5114 entries, 0 to 5113\n",
      "Data columns (total 9 columns):\n",
      " #   Column               Non-Null Count  Dtype  \n",
      "---  ------               --------------  -----  \n",
      " 0   shipment_id_1        5114 non-null   int64  \n",
      " 1   drop_off_point       5114 non-null   int64  \n",
      " 2   destination_country  5114 non-null   int64  \n",
      " 3   freight_cost         5114 non-null   float64\n",
      " 4   gross_weight         5114 non-null   float64\n",
      " 5   shipment_charges     5114 non-null   float64\n",
      " 6   shipment_mode        5114 non-null   int64  \n",
      " 7   shipping_company     5114 non-null   int64  \n",
      " 8   shipping_time        5114 non-null   float64\n",
      "dtypes: float64(4), int64(5)\n",
      "memory usage: 359.7 KB\n"
     ]
    }
   ],
   "source": [
    "shipping = pd.read_csv(\"shipping_train_2.csv\")\n",
    "shipping.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "29678e9e",
   "metadata": {},
   "source": [
    "### Train-test Splitting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "11115538",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Rows in train set : 4091\n",
      "Rows in test set : 1023\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "train_set , test_set = train_test_split(shipping, test_size=0.2, random_state=42)\n",
    "print(\"Rows in train set :\" , len(train_set))\n",
    "print(\"Rows in test set :\" , len(test_set))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8ddd7443",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import StratifiedShuffleSplit\n",
    "split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)\n",
    "for train_index, test_index in split.split(shipping, shipping['shipment_mode']):\n",
    "    strat_train_set = shipping.loc[train_index]\n",
    "    strat_test_set = shipping.loc[test_index]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7982a8b2",
   "metadata": {},
   "source": [
    "### Looking for correlations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "c6306e5b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "shipping_time          1.000000\n",
       "destination_country    0.260984\n",
       "gross_weight           0.208362\n",
       "shipment_id_1          0.197895\n",
       "freight_cost           0.172070\n",
       "shipment_charges       0.049514\n",
       "drop_off_point        -0.260984\n",
       "shipping_company      -0.662250\n",
       "shipment_mode         -0.754327\n",
       "Name: shipping_time, dtype: float64"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "corr_matrix =shipping.corr()\n",
    "corr_matrix.head()\n",
    "corr_matrix['shipping_time'].sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "04862547",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>shipment_id_1</th>\n",
       "      <th>drop_off_point</th>\n",
       "      <th>destination_country</th>\n",
       "      <th>freight_cost</th>\n",
       "      <th>gross_weight</th>\n",
       "      <th>shipment_charges</th>\n",
       "      <th>shipment_mode</th>\n",
       "      <th>shipping_company</th>\n",
       "      <th>shipping_time</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1451</th>\n",
       "      <td>329736</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>91.35</td>\n",
       "      <td>170.0</td>\n",
       "      <td>0.90</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10.46817</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>162</th>\n",
       "      <td>36738</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>93.70</td>\n",
       "      <td>385.0</td>\n",
       "      <td>0.90</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>34.57269</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3169</th>\n",
       "      <td>721720</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>88.15</td>\n",
       "      <td>1256.0</td>\n",
       "      <td>0.75</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>5.38241</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2575</th>\n",
       "      <td>584737</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>94.40</td>\n",
       "      <td>2000.0</td>\n",
       "      <td>0.90</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>20.64248</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3858</th>\n",
       "      <td>884705</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>91.73</td>\n",
       "      <td>1106.0</td>\n",
       "      <td>0.90</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>30.22674</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      shipment_id_1  drop_off_point  destination_country  freight_cost  \\\n",
       "1451         329736               1                    0         91.35   \n",
       "162           36738               1                    0         93.70   \n",
       "3169         721720               1                    0         88.15   \n",
       "2575         584737               1                    0         94.40   \n",
       "3858         884705               1                    0         91.73   \n",
       "\n",
       "      gross_weight  shipment_charges  shipment_mode  shipping_company  \\\n",
       "1451         170.0              0.90              0                 1   \n",
       "162          385.0              0.90              0                 1   \n",
       "3169        1256.0              0.75              1                 3   \n",
       "2575        2000.0              0.90              0                 1   \n",
       "3858        1106.0              0.90              0                 1   \n",
       "\n",
       "      shipping_time  \n",
       "1451       10.46817  \n",
       "162        34.57269  \n",
       "3169        5.38241  \n",
       "2575       20.64248  \n",
       "3858       30.22674  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "strat_train_set.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4ab37c8c",
   "metadata": {},
   "source": [
    "%matplotlib inline\n",
    "from pandas.plotting import scatter_matrix\n",
    "shipping.plot(kind=\"scatter\",x='shipment_mode',y='shipping_time', alpha=0.8)\n",
    "shipping.plot(kind=\"scatter\",x='shipping_company',y='shipping_time', alpha=0.8)\n",
    "shipping.plot(kind=\"scatter\",x='drop_off_point',y='shipping_time', alpha=0.8)clf.score(X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "6b46c3b9",
   "metadata": {},
   "outputs": [],
   "source": [
    "shipping = strat_train_set.drop(\"shipping_time\",axis=1)\n",
    "shipping_labels = strat_train_set[\"shipping_time\"].copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "4fafee4d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>shipment_id_1</th>\n",
       "      <th>drop_off_point</th>\n",
       "      <th>destination_country</th>\n",
       "      <th>freight_cost</th>\n",
       "      <th>gross_weight</th>\n",
       "      <th>shipment_charges</th>\n",
       "      <th>shipment_mode</th>\n",
       "      <th>shipping_company</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1451</th>\n",
       "      <td>329736</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>91.35</td>\n",
       "      <td>170.0</td>\n",
       "      <td>0.90</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>162</th>\n",
       "      <td>36738</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>93.70</td>\n",
       "      <td>385.0</td>\n",
       "      <td>0.90</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3169</th>\n",
       "      <td>721720</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>88.15</td>\n",
       "      <td>1256.0</td>\n",
       "      <td>0.75</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2575</th>\n",
       "      <td>584737</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>94.40</td>\n",
       "      <td>2000.0</td>\n",
       "      <td>0.90</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3858</th>\n",
       "      <td>884705</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>91.73</td>\n",
       "      <td>1106.0</td>\n",
       "      <td>0.90</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      shipment_id_1  drop_off_point  destination_country  freight_cost  \\\n",
       "1451         329736               1                    0         91.35   \n",
       "162           36738               1                    0         93.70   \n",
       "3169         721720               1                    0         88.15   \n",
       "2575         584737               1                    0         94.40   \n",
       "3858         884705               1                    0         91.73   \n",
       "\n",
       "      gross_weight  shipment_charges  shipment_mode  shipping_company  \n",
       "1451         170.0              0.90              0                 1  \n",
       "162          385.0              0.90              0                 1  \n",
       "3169        1256.0              0.75              1                 3  \n",
       "2575        2000.0              0.90              0                 1  \n",
       "3858        1106.0              0.90              0                 1  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "shipping.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "df077e70",
   "metadata": {},
   "source": [
    "### Creating Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "50638786",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.pipeline import Pipeline\n",
    "\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "my_pipeline = Pipeline([\n",
    "    ('std_scaler',StandardScaler())\n",
    "    ])\n",
    "shipping_tr = my_pipeline.fit_transform(shipping)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cb2f0b54",
   "metadata": {},
   "source": [
    "### Selecting a desired model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "8b22c1f1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestRegressor()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestRegressor</label><div class=\"sk-toggleable__content\"><pre>RandomForestRegressor()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestRegressor()"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestRegressor\n",
    "model = RandomForestRegressor()\n",
    "model.fit(shipping_tr,shipping_labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "eb23e88b",
   "metadata": {},
   "outputs": [],
   "source": [
    "shipping_tr = pd.DataFrame(shipping_tr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "294a0fc1",
   "metadata": {},
   "outputs": [],
   "source": [
    "some_data = shipping_tr.iloc[:50]\n",
    "some_labels = shipping_labels.iloc[:50]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "b447ca98",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\shubh\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\sklearn\\base.py:450: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([20.1747584, 20.1747584,  5.1865117, 20.1747584, 20.1747584,\n",
       "        5.1893785,  5.1865117,  5.1865117, 21.4325582, 20.1747584,\n",
       "       20.1747584, 20.1747584,  5.1893785, 20.1747584,  5.1893785,\n",
       "        5.1865117, 20.1747584,  5.1893785, 20.1747584, 20.1747584,\n",
       "       20.1747584,  5.1865117,  5.1865117,  5.1865117, 20.1747584,\n",
       "       20.1747584,  5.1865117,  5.1865117,  5.1893785, 20.1747584,\n",
       "        5.1865117, 20.1747584,  5.1865117,  5.1893785,  5.1893785,\n",
       "        5.1865117, 20.1747584, 20.1747584,  5.1865117,  5.1865117,\n",
       "       20.1747584,  5.1865117, 20.1747584,  5.1865117, 20.1747584,\n",
       "        5.1865117, 20.1747584, 20.1747584,  5.1865117,  5.1865117])"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prepared_data = my_pipeline.transform(some_data)\n",
    "model.predict(prepared_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "f2cea1b3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[10.46817,\n",
       " 34.57269,\n",
       " 5.38241,\n",
       " 20.64248,\n",
       " 30.22674,\n",
       " 5.1169,\n",
       " 5.05104,\n",
       " 5.10162,\n",
       " 12.82431,\n",
       " 12.93565,\n",
       " 23.4331,\n",
       " 21.08553,\n",
       " 5.02917,\n",
       " 17.7434,\n",
       " 5.20405,\n",
       " 5.0353,\n",
       " 32.1809,\n",
       " 5.25266,\n",
       " 30.73623,\n",
       " 10.49005,\n",
       " 30.08588,\n",
       " 5.07106,\n",
       " 5.2044,\n",
       " 5.01146,\n",
       " 34.14479,\n",
       " 9.17373,\n",
       " 5.21007,\n",
       " 5.30127,\n",
       " 5.19676,\n",
       " 11.09502,\n",
       " 5.20903,\n",
       " 29.58264,\n",
       " 5.14201,\n",
       " 5.15498,\n",
       " 5.15023,\n",
       " 5.17025,\n",
       " 31.23044,\n",
       " 28.98935,\n",
       " 5.24086,\n",
       " 5.15671,\n",
       " 26.31516,\n",
       " 5.36678,\n",
       " 8.92604,\n",
       " 5.10035,\n",
       " 18.47778,\n",
       " 5.14444,\n",
       " 7.83819,\n",
       " 22.68102,\n",
       " 5.35162,\n",
       " 5.14236]"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "list(some_labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "e9047b1e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\shubh\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\sklearn\\base.py:443: UserWarning: X has feature names, but RandomForestRegressor was fitted without feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "158.62460373048702"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import mean_squared_error\n",
    "shipping_predictions = model.predict(shipping)\n",
    "rf_mse = mean_squared_error(shipping_labels, shipping_predictions)\n",
    "rf_rmse = np.sqrt(rf_mse)\n",
    "rf_mse"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "342b1daa",
   "metadata": {},
   "source": [
    "### Using better evaluation technique - Cross Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "a86a7912",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([6.86231979, 7.30217389, 6.76817341, 7.38957842, 7.440991  ,\n",
       "       6.70286504, 6.9758928 , 6.89813602, 7.34772293, 7.5178796 ])"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "scores = cross_val_score(model, shipping, shipping_labels, scoring= \"neg_mean_squared_error\",cv=10)\n",
    "rmse_scores = np.sqrt(-scores)\n",
    "rmse_scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "73bdc9c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "def print_scores(scores):\n",
    "    print(\"scores:\",scores)\n",
    "    print(\"mean:\", scores.mean())\n",
    "    print(\"Standard deviation: \",scores.std() )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "40a9a50c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "scores: [6.86231979 7.30217389 6.76817341 7.38957842 7.440991   6.70286504\n",
      " 6.9758928  6.89813602 7.34772293 7.5178796 ]\n",
      "mean: 7.120573289915834\n",
      "Standard deviation:  0.29210114944629223\n"
     ]
    }
   ],
   "source": [
    "print_scores(rmse_scores)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c6d3b70e",
   "metadata": {},
   "source": [
    "## Saving the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "38a61c4c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Assignment.joblib']"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from joblib import load,dump\n",
    "dump(model, 'Assignment.joblib')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2e0cc11f",
   "metadata": {},
   "source": [
    "## Testing the model on Test data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "aba8f72a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7.358459449443684"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test = strat_test_set.drop('shipping_time',axis=1)\n",
    "Y_test = strat_test_set[\"shipping_time\"].copy()\n",
    "X_test_prepared = my_pipeline.transform(X_test)\n",
    "final_predictions = model.predict(X_test_prepared)\n",
    "final_mse = mean_squared_error(Y_test,final_predictions)\n",
    "final_rmse = np.sqrt(final_mse)\n",
    "final_rmse\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5e4eb2df",
   "metadata": {},
   "source": [
    "## Final result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "641d5f73",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>shipment_id_1</th>\n",
       "      <th>drop_off_point</th>\n",
       "      <th>destination_country</th>\n",
       "      <th>freight_cost</th>\n",
       "      <th>gross_weight</th>\n",
       "      <th>shipment_charges</th>\n",
       "      <th>shipment_mode</th>\n",
       "      <th>shipping_company</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2736</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>86.81</td>\n",
       "      <td>100.0</td>\n",
       "      <td>0.75</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2738</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>94.43</td>\n",
       "      <td>1006.0</td>\n",
       "      <td>0.75</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5739</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>93.55</td>\n",
       "      <td>321.0</td>\n",
       "      <td>1.05</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8722</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>88.74</td>\n",
       "      <td>355.0</td>\n",
       "      <td>1.05</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>9737</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>92.83</td>\n",
       "      <td>115.0</td>\n",
       "      <td>1.05</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   shipment_id_1  drop_off_point  destination_country  freight_cost  \\\n",
       "0           2736               1                    0         86.81   \n",
       "1           2738               1                    0         94.43   \n",
       "2           5739               1                    0         93.55   \n",
       "3           8722               1                    0         88.74   \n",
       "4           9737               1                    0         92.83   \n",
       "\n",
       "   gross_weight  shipment_charges  shipment_mode  shipping_company  \n",
       "0         100.0              0.75              1                 3  \n",
       "1        1006.0              0.75              1                 3  \n",
       "2         321.0              1.05              1                 2  \n",
       "3         355.0              1.05              1                 2  \n",
       "4         115.0              1.05              1                 2  "
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "strat_test_set_1 = pd.read_csv(\"ASSIGNMENT_TESTING.csv\")\n",
    "strat_test_set_1.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "9c101b48",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7.358459449443684"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test_prepared_final = my_pipeline.transform(strat_test_set_1)\n",
    "final_predictions_1 = model.predict(X_test_prepared_final)\n",
    "final_rmse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "ec376c68",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 5.1346732  5.1624878  5.241534  ... 16.4791497 14.0907585 17.804818 ]\n"
     ]
    }
   ],
   "source": [
    "print(final_predictions_1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "94b1707d",
   "metadata": {},
   "outputs": [],
   "source": [
    "final_predictions_1 = pd.DataFrame(final_predictions_1)\n",
    "final_predictions_1.to_csv('result.csv', index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "89c0c56f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9d6c2c1b",
   "metadata": {},
   "outputs": [],
   "source": [
    "tst = pd.read_csv('result.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8b7169b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "tst_c = tst['0']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "0108b60b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0        5.134673\n",
       "1        5.162488\n",
       "2        5.241534\n",
       "3        5.223931\n",
       "4        5.236355\n",
       "          ...    \n",
       "1255    21.263308\n",
       "1256    17.281983\n",
       "1257    16.479150\n",
       "1258    14.090759\n",
       "1259    17.804818\n",
       "Name: 0, Length: 1260, dtype: float64"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tst_c"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b9f63a87",
   "metadata": {},
   "outputs": [],
   "source": [
    "tst2  = pd.read_csv('assignment_result.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "442a2e15",
   "metadata": {},
   "outputs": [],
   "source": [
    "tst2['shipping_time'] = tst_c"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
