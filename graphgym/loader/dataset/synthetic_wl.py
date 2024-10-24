import os.path as osp
import shutil
from pathlib import Path
from typing import Callable, List, Optional

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from tqdm import trange
import scipy.io as scio

LOCAL_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


class SyntheticWL(InMemoryDataset):
    r"""Synthetic graphs dataset collected from https://arxiv.org/abs/2010.01179
    and https://arxiv.org/abs/2212.13350.

    Supported datasets:

        - EXP
        - CEXP
        - SR25

    """

    _supported_datasets: List[str] = [
        "exp",
        "cexp",
        "sr25",
        "tri",
        "trix",
        "cc",
        "cg"
    ]

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        extrapolate=False,
    ):
        if name.startswith("cc") or name.startswith("cg"):
            target = int(name[2:])
            name = name[:2]
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_dir)
        print(self.processed_paths)
        if extrapolate:
            self.data, self.slices = torch.load(self.processed_paths[1])
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])
            if self._name == "trix":
                self.test = SyntheticWL(root, name, transform, pre_transform, pre_filter, extrapolate=True)
        if name.startswith("cc") or name.startswith("cg"):
            self.split_idxs = SC_indices
            y_train_val = torch.cat([self.data.y[self.split_idxs[0]], self.data.y[self.split_idxs[1]]], dim=0)
            mean = y_train_val.mean(dim=0)
            std = y_train_val.std(dim=0)
            self.data.y = (self.data.y-mean)/std
            self.data.y = self.data.y[:, target]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self):,}, name={self.name!r})"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        if not isinstance(name, str):
            raise TypeError(f"Name must be a string, got {type(name)}")
        name = name.lower()
        if name in ["exp", "cexp"]:
            self.download = self._download_exp
            self._process_data_list = self._process_data_list_exp
            self._raw_file_names = ["GRAPHSAT.txt"]
        elif name == "sr25":
            self.download = self._download_sr25
            self._process_data_list = self._process_data_list_sr25
            self._raw_file_names = ["sr251256.g6"]
        elif name == "tri" or name == "trix":
            self._process_data_list = self._process_data_list_tri
        elif name.startswith("cc") or name.startswith("cg"):
            self._process_data_list = self._process_data_list_SC
            self._raw_file_names = ["data.mat"]
        else:
            raise ValueError(f"Unrecognized dataset name {name!r}, available "
                             f"options are: {self._supported_datasets}")
        self._name = name

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.__class__.__name__, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.__class__.__name__, self.name,
                        "processed")

    @property
    def raw_file_names(self) -> List[str]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> str:
        return ["data.pt", "test.pt"]

    def process(self):
        data_list = self._process_data_list()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print(f"{self.processed_paths[0]=}")

        torch.save(self.collate(data_list), self.processed_paths[0])

        if self._name == "trix":
            data_list = self._process_data_list(num_nodes=100)

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            torch.save(self.collate(data_list), self.processed_paths[1])

    def _download_exp(self):
        filename = self.raw_file_names[0]
        data_path = LOCAL_DATA_DIR / "Abboud2020" / self.name.upper() / filename
        shutil.copyfile(data_path, self.raw_paths[0])

    def _process_data_list_exp(self):
        data_list = []
        with open(self.raw_paths[0]) as f:
            # First line is an integer indicating the total number of graphs
            num_graphs = int(f.readline().rstrip())
            for _ in trange(num_graphs):
                # First line of each block: num_nodes, graph_label
                num_nodes, label = map(int, f.readline().rstrip().split(" "))
                adj = np.zeros((num_nodes, num_nodes))
                x = np.zeros((num_nodes, 1), dtype=int)

                for src, line in zip(range(num_nodes), f):
                    values = list(map(int, line.rstrip().split(" ")))
                    x[src] = values[0]

                    for dst in values[2:]:
                        adj[src, dst] = 1
                edge_index = np.vstack(np.nonzero(adj))

                data = Data(x=torch.LongTensor(x),
                            edge_index=torch.LongTensor(edge_index),
                            y=torch.LongTensor([label]))
                data_list.append(data)

        return data_list

    def _download_sr25(self):
        url = "https://raw.githubusercontent.com/XiaoxinHe/Graph-MLPMixer/48cd68f9e92a7ecbf15aea0baf22f6f338b2030e/dataset/sr25/raw/sr251256.g6"
        download_url(url, self.raw_dir)

    def _process_data_list_sr25(self):
        data_list = []
        for i, g in enumerate(nx.read_graph6(self.raw_paths[0])):
            adj_coo = nx.to_scipy_sparse_array(g).tocoo()
            x = torch.ones(g.size(), 1)
            edge_index = torch.LongTensor(np.vstack((adj_coo.row, adj_coo.col)))
            y = torch.LongTensor([i])
            data_list.append(Data(x=x, edge_index=edge_index, y=y))
        return data_list

    def _process_data_list_tri(self, num_nodes=20):
        return generate_triangle_graphs(num_nodes=num_nodes)

    def _process_data_list_SC(self):
        # process npy data into pyg.Data
        print('Processing data from ' + self.raw_dir + '...')
        raw_data = scio.loadmat(self.raw_paths[0])
        data_list = [self.adj2data(raw_data['A'][0][i], raw_data['F'][0][i]) for i in range(5000)]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            temp = []
            for i, data in enumerate(data_list):
                if i % 100 == 0:
                    print('Pre-processing %d/%d' % (i, len(data_list)))
                temp.append(self.pre_transform(data))
            data_list = temp
        return data_list

    def adj2data(self, A, y):
        begin, end = np.where(A == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        if y.ndim == 1:
            y = y.reshape([1, -1])
        x = torch.ones( A.shape[0], 1)
        return Data(edge_index=edge_index, y=torch.tensor(y), x=x)


def generate_triangle_graphs(num_graphs=1000, num_nodes=20):
    data_set = []
    for _ in range(num_graphs):
        G = nx.random_regular_graph(3, num_nodes)
        dict = nx.triangles(G)
        edge_index = [[],[]]
        labels = [0]*num_nodes
        for u, v, _ in G.edges(data=True):
            edge_index[0] += [u, v]
            edge_index[1] += [v, u]
        for key, value in dict.items():
            labels[key] = int(value>0)
        data = Data(edge_index=torch.tensor(edge_index), y=torch.tensor(labels))
        data.num_nodes = num_nodes
        data_set += [data]
    return data_set


SC_indices = [[398,3833,4836,4572,636,2545,1161,2230,148,2530,4070,1261,4682,333,906,3170,483,2825,1778,2466,159,1563,
               402,4258,4775,1095,3054,4268,3711,453,4085,3861,4390,1902,2412,1362,3777,3589,3897,3411,3635,2550,4643,
               1571,608,2172,675,4175,376,3009,3901,3740,134,1654,1875,4344,533,4763,2219,2876,2111,933,2636,3048,4394,
               464,2048,2796,4523,2036,381,3936,862,3696,567,1281,1002,138,3625,3222,2886,308,3528,304,1270,817,1098,
               1329,1965,3583,3014,690,4178,502,291,2929,2814,4214,2831,1139,949,1379,4787,4102,891,3774,3682,1526,4185,
               1835,113,4257,3594,1465,1450,2513,4284,3812,4371,2587,2027,4315,1470,4348,4544,3846,214,2465,4834,1351,
               678,4863,3052,939,520,3650,1974,4467,2154,3288,2216,142,3488,4130,2329,1702,4120,3733,3865,3443,405,3655,
               4169,4150,298,735,4672,2370,42,4244,624,4583,3595,666,1793,3001,4995,4612,4445,1109,2247,4248,4137,562,
               2452,2707,3830,2782,4170,1680,1122,49,1091,137,3098,302,3362,12,4540,2631,4098,4694,130,1141,30,2776,
               4506,1803,4363,977,871,3794,3270,202,2038,575,4030,2445,3801,1511,3554,2951,2864,3881,704,4463,125,3580,
               2378,154,4837,3380,2030,3464,601,1237,3348,98,3993,1490,3577,39,3416,1741,4872,348,2531,72,867,812,4345,
               3664,4323,1185,4843,3201,3507,1670,4468,1343,491,495,2091,3779,3239,3575,3539,2816,4847,2616,3606,4171,
               1658,1772,4580,1852,499,1544,4452,849,2807,2686,4328,2640,3634,2118,716,158,4166,1005,124,4204,4121,3177,
               2232,2939,530,776,4391,2239,3453,2815,3217,918,1328,446,4953,1377,3899,1259,4844,3743,842,3866,3731,241,
               2839,2985,4979,771,3824,3876,1011,843,4700,1545,192,3735,2872,1338,3343,2517,3689,3354,380,29,4243,3672,
               3771,1051,534,1267,3274,4533,3503,4510,2947,1116,1969,4283,3307,4617,3923,4522,2658,1115,3892,2731,1949,
               4529,4961,3970,1765,347,4613,3188,2590,3845,4441,1419,1800,1399,1705,661,1977,4982,4379,4594,1428,4727,
               4932,1620,3859,2417,942,4610,3152,2087,2166,3089,489,4963,1839,4602,4581,3317,2519,4074,1114,2744,349,
               4287,1991,3342,3141,1012,352,896,799,2455,4474,2788,1957,2473,639,1487,3161,2812,2592,3566,1082,4256,
               2383,531,541,4902,825,3088,4413,1738,2701,314,4830,2848,2214,1387,2201,2630,3326,1366,2049,394,4667,3315,
               2315,2311,2849,3233,1367,1246,3608,4429,2177,4398,4173,4954,1934,4409,2070,727,2379,2075,3199,3864,487,
               4624,780,3891,3341,3563,3228,218,4912,44,683,311,3438,2840,2182,3425,31,1418,1749,1108,3839,1567,4590,
               1022,2263,452,4899,2611,282,651,4046,1326,1176,2500,582,2875,2982,4213,4596,1393,3390,1928,4366,3550,
               2189,3480,3871,4299,4719,3195,1922,1075,751,463,4076,725,4785,2108,2164,3961,2752,245,1085,4731,4156,
               2942,3103,2769,4304,2838,4225,4347,2626,1712,599,1014,3699,500,1480,3247,4711,3631,578,2131,4928,2523,
               3320,873,556,2457,2779,2667,1065,470,3959,2879,3889,2217,1330,34,4134,3196,4952,895,4942,2959,2400,1160,
               3435,2553,4500,104,3712,3151,2808,4911,1358,1921,3925,2394,2926,1531,634,3303,95,4586,1119,4929,1223,
               2527,1491,883,2664,415,4080,2625,3814,224,2992,1586,1783,3673,3174,3281,2533,2716,2062,2494,3253,3129,
               965,269,861,3624,3932,2332,507,1912,4251,2284,117,1349,1551,2013,3872,170,383,2703,4577,1894,773,4235,
               3994,4104,4295,4094,2350,2364,3021,2702,4146,1666,3654,1946,1675,962,4012,1319,619,1906,872,223,3010,
               2944,1525,584,2727,4160,866,4674,496,831,2149,189,173,4976,4367,2762,4458,2104,3318,1787,3414,2470,4346,
               4568,3002,4239,1995,628,4562,410,1710,2854,1916,1475,2612,3061,1997,1309,3976,2980,2692,4208,1371,1522,
               4062,3450,4783,1652,220,2382,2758,2655,1625,3562,1564,2781,1879,4815,2924,3535,3776,3600,4885,1824,1423,
               33,2819,379,517,864,1001,3621,481,3739,3640,1087,1042,3428,4286,4563,162,2907,2909,3807,45,731,2459,4817,
               4546,3506,2768,2601,1986,4592,1299,2347,2678,3173,2714,3684,3213,1669,2644,4202,3620,2794,2994,2231,548,
               2617,1495,1084,2672,1203,3169,3651,976,4718,2365,1752,4427,629,1774,4582,4007,2228,2372,1411,3426,1754,
               910,3023,4656,3135,1437,4471,3758,4356,3360,1239,3803,538,2620,613,2889,3206,4828,1519,543,4531,3035,
               1590,1250,1643,4738,1180,991,806,4576,2803,4806,3312,1003,882,2974,1951,2493,4634,668,1043,4499,2358,
               4481,2946,2730,4478,2778,4003,3230,3710,4113,2682,4096,135,4933,526,2255,1657,979,641,2102,3985,1892,
               2764,2565,4334,2600,1521,3847,1507,4002,4918,1431,2510,4475,1831,4569,70,1224,728,3046,1530,3906,2726,
               2605,472,4644,660,57,2034,1412,4217,246,3184,1789,3166,1275,1478,271,898,3357,2293,294,4095,3548,4642,
               4938,2923,195,514,3072,4559,971,3429,1172,4875,406,1784,32,2307,854,4537,4269,4266,2100,1695,2917,3617,
               2088,2351,4008,2250,2777,521,4695,4520,4756,378,1385,4457,2905,3500,1389,3364,9,1013,1499,1066,3701,1569,
               4052,4302,1688,3118,1461,2676,253,4650,3700,1697,1430,1870,720,938,122,4866,2483,1069,829,87,3934,528,
               2809,2453,2024,161,4840,1248,545,1596,1460,3957,3240,4664,263,2618,1853,3791,878,1540,1960,890,852,4879,
               3848,4073,1368,1414,847,1819,684,396,2866,2011,3053,3407,401,1606,1262,1529,1850,2501,4638,3704,559,4115,
               2805,3873,1391,4228,2367,788,1452,3265,2913,3275,775,217,1245,4316,4619,1363,630,3572,4124,4101,1881,
               1482,156,1196,2824,3283,3809,3908,2006,4740,2490,4139,22,4196,4054,2799,3578,4722,4372,3376,3662,4716,
               3780,1715,2377,3805,4993,3755,1900,4881,2112,326,252,1290,3355,2137,2439,4114,1448,2200,167,1940,844,
               4172,1509,2976,2829,4483,888,4530,4842,3565,4630,2337,1708,1177,4502,1958,581,3144,3785,3051,3707,4111,
               2065,1996,4654,4203,2755,2719,2069,4916,2619,3518,980,2606,2491,1533,3697,3920,3099,2991,2285,4628,1891,
               1814,230,4709,283,1696,2915,2666,3484,4018,4743,4200,3659,4083,3245,2212,4223,88,4388,2303,1380,1556,
               4277,648,1632,1298,995,1420,3570,4925,1118,2983,2224,3761,3576,667,1706,4300,1829,3680,1272,2407,2647,
               4960,2196,900,1979,3194,4016,2615,3910,1332,1505,2373,1364,2106,1104,325,345,4512,1750,23,1,2873,3493,
               708,3350,4922,4013,555,1766,4692,2160,4199,4159,2345,4945,2044,2260,1327,4761,3434,982,1855,4717,1410,
               4215,4712,927,3335,1790,480,2954,2223,803,2724,536,2243,3388,3025,4396,4194,3473,3156,2447,2275,3491,
               4770,4939,4349,4418,1953,4748,4411,638,185,3381,2888,680,2124,4132,3432,485,4955,4072,1761,4802,4143,
               1220,3332,4118,462,3162,3628,4093,2140,4848,3956,2188,1826,553,2828,4181,1767,3119,3481,2487,4796,286,
               465,2670,37,3081,2287,4154,4820,1202,3759,622,4365,1457,2058,2078,1536,1642,2215,2863,3349,3374,4689,
               3287,3687,408,4753,1376,1077,1515,1926,4708,2669,4598,1256,429,4894,3058,2033,3482,2338,4231,1127,2962,
               1489,2718,2953,917,1432,3008,2958,1163,1999,4560,3983,4940,85,4501,248,519,2743,4065,712,1277,4400,1662,
               3637,3065,809,2381,3519,501,3229,227,2125,3477,3180,1068,3095,1651,2736,620,621,4446,2012,3286,3305,4157,
               3319,2081,748,1763,4292,4547,685,4240,4079,221,1433,203,1009,36,4161,4326,4313,3569,3585,1992,1271,1021,
               2397,4144,1726,1232,4791,1474,2460,576,836,1722,4298,3986,569,2865,2278,1622,182,382,4493,3238,3804,1615,
               4237,706,1888,4686,1736,206,1759,384,2753,2134,1755,4168,3311,457,4036,3289,3629,2306,3905,2150,2016,
               4183,663,2635,3392,1547,3553,2339,1692,1471,768,3235,3967,4209,2076,3371,3086,1861,1813,1640,4936,3561,
               3108,4495,841,4989,1815,4490,3648,3517,3726,3783,4659,857,2009,2795,1303,925,2846,1771,1742,3987,2175,
               1422,4551,2585,4188,4399,3082,2472,3960,1497,1889,1273,1288,4165,1834,4440,3077,3887,4744,2878,4691,687,
               1907,3226,3267,3011,3996,2574,2281,527,3282,1230,2971,4460,1867,2748,794,2165,3489,1493,1982,1804,4005,
               4179,3236,2760,853,787,3963,4386,4609,1523,1718,2324,1324,546,615,369,4554,4807,1073,2906,1357],
              [2858,1559,1441,2179,1390,2575,467,4448,276,1046,4535,2270,2832,4320,3883,1514,1170,1837,1359,2318,1746,
               3405,3027,1280,2060,4086,3225,1047,4404,4282,3770,2604,3045,3784,2089,4152,2551,53,516,3440,1757,3840,
               985,4021,3964,1370,3706,482,2722,3564,2017,1593,4924,4097,2693,4389,692,2566,289,3389,1729,2885,2729,
               3029,2272,3515,1581,295,61,1580,3016,3746,2475,1698,2171,2649,2709,4649,4314,4811,2648,3304,3643,3461,
               3340,589,1961,1464,590,4486,1144,3266,3327,3256,2323,2497,3714,3992,4833,2409,2749,607,296,587,3394,566,
               1687,2505,3826,2650,3760,934,2931,3521,2067,1791,196,1760,133,2586,2343,321,118,261,598,2265,905,4254,
               2949,3400,3984,1056,2584,322,1833,4585,149,4527,2728,579,421,4671,4001,4281,3504,1691,2052,1190,1266,
               2294,2826,4871,226,1721,17,3472,1933,2577,2608,2451,2071,1206,4004,418,1939,988,1062,3276,3688,2204,572,
               229,4058,86,2613,440,1865,1427,2869,288,3657,2346,1300,2771,242,3930,3895,1382,422,4981,3132,2797,4455,
               2938,3183,1182,998,4126,832,3140,944,935,2591,3200,2629,1292,880,4778,996,4896,346,558,4466,2507,4889,
               695,3223,1504,257,2178,2884,3903,1148,1869,2902,4189,2211,3532,1887,3127,953,471,2966,1990,3757,2948,
               3490,3124,2921,3220,840,4319,1150,3862,1513,1458,1941,4882,1984,1445,2892,1621,1649,92,3214,1700,1828,
               1931,2717,1566,234,3171,4860,897,3944,2639,961,1285,3100,983,1336,3007,3302,3479,1269,4973,2162,4984,
               2549,3598,2448,1120,3147,2368,450,635,2406,444,2152,338,868,1830,3950,2539,389,1972,1473,436,4312,3251,5,
               3133,3831,4857,4410,109,1034,3496,368,3012,834,2571,1607,1283,3685,443,2148,4970,2645,4556,3870,643,570,
               451,1197,1070,3055,2072,2369,2477,4930,2998,2184,4149,3321,3,2357,4318,989,306,10,14,2103,2700,2063,4,
               682,2450,4838,1872,3529,332,2940,1282,285,4487,473,3904,2740,2359,4816,3197,2802,3681,1542,4331,2193,
               2933,3113,4222,2841,3263,4607,3524,2975,4948,3552,290,3763,729,2542,4962,3037,4449,1601,914,4825,249,
               4636,749,535,509,66,188,1794,219,1404,1242,1836,244,835,3377,3092,1276,2537,4414,2327,1129,596,1284,758,
               654,1496,15,4751,4698,2374,1936,879,2770,4891,4923,2290,144,388,4701,4593,3386,2920,4865,4031,2240,2793,
               1877,1503,4536,4028,1174,3558,688,317,1704,4301,3338,4819,2817,1644,2277,4771,1018,1436,177,50,958,64,
               2308,4595,3373,4846,4561,3237,3773,1573,1191,4303,399,2433,2901,3658,3951,2314,3693,3720,2602,2122,2234,
               461,4941,4357,2301,4280,2543,4625,1812,2480,3297,899,2981,386,4776,305,2526,3075,409,1451,3138,191,3294,
               3073,361,200,3832,4927,4088,3644,3323,2203,565,1386,4162,4818,1744,426,2340,3485,4710,929,2498,3260,4294,
               4755,4549,1240,3649,4647,2404,1287,2804,1574,1989,2955,1373,959,1768,71,390,3385,2696,1334,169,743,2830,
               2066,951,4849,190,4699,1342,1895,97,2047,2425,1183,3931,4434,1577,4805,4589,4798,1093,4117,316,2396,132,
               1770,1494,3949,1667,299,616,2300,4133,1059,3448,2509,4538,2242,3882,703,1128,2194,3636,4597,414,3495,211,
               3998,3030,4082,4497,1646,940,4980,3844,713,1078,4206,3694,1955,2528,779,796,3210,4508,2385,936,351,3768,
               1733,365,2244,3314,4990,1015,1081,2355,2481,744,686,963,677,3916,3509,1719,2855,1235,3666,4437,2141,3422,
               4632,1236,1037,4325,2623,3391,3534,458,3379,955,924,3817,2440,574,3452,719,1195,1074,4412,4855,3990,4022,
               3039,1553,3827,3040,1032,1856,3977,547,793,1501,1572,3442,330,4637,2579,1713,4442,2747,2882,710,3036,
               1101,4861,4017,356,3456,493,3813,69,1313,4337,4766,1618,3079,1535,175,981,4274,826,1254,3244,2010,2401,
               3076,764,4800,1600,2,800,786,2961,4919,674,2720,2706,4255,2534,238,427,2423,204,923,240,2443,1927,3471,
               670,3397,4354,694,1023,3439,3080,1044,366,4758,1140,3880,259,3074,3546,3875,1214,163,4831,1583,264,385,
               1943,3556,3255,4987,701,4488,3675,272,2734,3540,1395,277,1151,3668,1886,1446,4489,4306,3031,2143,1094,
               4184,4498,2868,964,4127,179,3044,4382,2594,4541,1952,2157,105,999,488,1341,3885,2041,2139,3111,4310,2019,
               4153,3787,1602,1157,4364,1052,1149,1762,2735,3900,1994,3207,1225,68,2042,1555,4340,1660,2464,1548,4601,
               975,76,2525,1560,3581,4876,2004,4684,1899,2518,2552,1948,1255,1728,2436,3043,2057,3601,1911,3894,618,
               1310,4158,2117,4027,4745,3512,4893,846,4465,4621,300,4542,1732,51,3145,874,2299,2478,4884,2110,3786,4059,
               2580,2663,3941,1898,1079,850,228,2085,4769,3227,4959,2996,1914,1699,1048,235,1751,3718,2039,1347,1811,
               297,2399,4063,2185,2421,353,3352,1775,3896,856,1917,2835,3915,4516,1554,2540,2471,3122,972,419,411,2384,
               2258,2101,4799,4897,791,1317,1693,1647,657,2031,4462,4868,313,943,3856,1656,4397,1036,1703,503,320,4706,
               2174,1311,954,4552,2484,4275,2146,4886,40,1832,4730,4034,2161,454,4824,722,4308,2155,2943,3542,4996,4558,
               3325,681,1100,4675,702,1825,1397,4772,1112,652,4270,4453,1110,3790,4048,3427,529,3126,4290,1388,413,937,
               1184,3120,3669,3284,1860,3462,711,3653,2857,3750,3884,1945,564,4514,2092,746,4804,655,2806,323,4278,4839,
               2899,1137,2784,425,2578,4335,1947,4296,3614,3652,600,1798,4821,3855,4428,1038,783,1268,4794,3869,2651,
               121],
              [778,984,3713,3160,745,4873,2916,1764,3732,2895,1171,2969,1425,176,2843,2761,82,805,2922,2094,1878,141,
               2000,2715,1318,1058,4784,3262,515,1199,2169,2310,2772,4665,3502,2254,838,1739,1194,232,4402,4151,2361,
               1355,1462,2183,1818,1424,2964,820,886,393,2790,1780,2462,3823,3424,476,2928,456,108,3212,762,3646,2198,
               789,4383,2319,11,1138,3209,3254,1278,3474,672,2486,1944,1905,2025,3351,3975,3514,700,3526,3368,669,2362,
               119,3034,274,3063,1396,4660,2972,4000,1633,4119,4661,215,1587,1231,392,27,3567,752,1810,2708,478,2818,
               3965,909,3110,3772,4385,4084,2485,4687,2151,2180,2331,3475,563,2759,921,4438,969,1212,3610,913,2657,3937,
               4690,2411,3981,723,2126,3067,2123,1027,597,6,1061,1449,4702,2661,997,4454,4492,4197,2689,653,3179,1467,
               3911,48,318,919,3328,1609,1950,3459,3359,340,4926,4803,3890,3097,3592,1215,2415,4077,3723,3647,518,1724,
               4575,2842,766,4406,1168,474,3375,3698,4322,268,1049,3497,2810,4641,4696,107,4715,2952,4473,1983,2573,
               4741,4246,4351,4431,1426,3736,3953,3955,2911,1694,2554,813,1476,1158,2558,2823,912,2252,3584,2535,3202,
               438,4965,609,2621,3725,3408,4177,1055,3501,1354,993,2095,4129,4883,2142,4850,1603,3149,1296,1808,4064,
               2967,3974,459,254,1920,303,4060,676,1550,2192,4618,3639,3619,41,4892,1549,1264,2900,865,3104,2570,1663,
               4864,2791,3852,4219,1041,1612,3019,2438,187,932,251,3403,357,278,1717,1463,2898,2280,1923,610,3979,1374,
               2694,328,781,3498,4683,1962,3633,4903,2984,2673,2476,3902,3175,2420,4774,2115,2713,3090,2560,4951,784,
               3234,4788,522,4100,3410,1210,1083,1017,1403,1723,3378,3754,1244,4136,4140,145,8,3667,3259,540,80,1978,
               2210,4233,3006,1106,1444,1846,2032,331,359,1915,486,1187,13,4869,3645,2427,3158,4450,1610,2226,757,1063,
               90,84,4532,4112,4220,966,4813,4106,4141,59,4236,1918,1976,3857,4241,4109,4376,2688,1673,4681,4964,4742,
               439,2488,477,4768,1117,715,2074,3574,2741,4697,2054,1372,4555,4571,3836,2756,3154,3536,391,2268,1238,58,
               568,3630,4395,3050,3838,2751,26,4792,3947,1897,3157,1293,4195,1173,1136,1626,717,1748,1025,2432,588,3022,
               1854,1477,4615,4901,4781,3522,3041,4015,2556,855,2986,4092,3663,1737,774,859,1785,3447,819,3538,4033,
               3083,455,52,1806,3842,20,3324,194,1213,4603,1086,1346,1217,2312,3638,3933,1454,3449,2341,3579,2627,3301,
               3792,2096,1756,4704,3921,1735,1809,2271,4247,585,4392,360,603,2877,1727,3954,3131,689,3038,1599,468,1315,
               1147,1686,2334,1222,1714,2080,4338,2145,1099,3412,4666,4679,795,4663,3150,4639,557,4888,4967,315,1604,
               926,2181,1035,3537,602,3626,1103,3322,1592,67,4822,157,2005,3853,1959,1980,4790,4895,1648,3241,344,1352,
               3530,442,3056,4336,2205,4765,1636,4526,505,3798,3815,18,4584,2248,646,3068,1930,772,47,3616,2856,4482,
               1929,3587,3938,3460,1485,4494,3071,1773,2774,3722,4944,2945,1743,2705,1637,3602,3433,1045,1211,1252,3059,
               1010,3499,1627,2461,2302,3005,2675,1260,4999,2348,1885,128,4814,1054,3568,110,1257,1105,2894,1925,2833,
               1541,2434,3789,2699,4669,2055,2853,1845,4913,4343,4249,2970,1301,2132,1685,466,3231,4024,1263,3469,2506,
               3060,4737,767,1394,1030,3816,1348,4988,2249,1857,1512,1709,3346,354,3066,2213,4293,1543,876,4890,2209,
               2597,2297,4670,3028,884,490,3353,4599,4777,3573,1142,822,492,665,4623,3446,990,327,4339,139,3431,2622,
               3738,250,342,4773,3764,3705,2390,792,1000,2628,441,2837,1302,4570,1314,2583,1019,916,43,3417,4507,2128,
               3172,270,4090,4676,511,475,532,4359,1323,893,2871,267,233,1286,2330,4900,2064,3788,287,3139,733,2950,512,
               810,3909,3888,3849,4035,1679,815,828,3559,279,1844,3487,2298,2403,3819,2419,3093,4545,1883,2079,166,2912,
               2891,3691,3278,2697,2890,4381,4368,3683,4992,2588,4633,1578,4190,309,1827,4271,3586,4311,1981,818,3382,
               1779,2896,3309,111,1539,1758,3719,3015,3677,1538,1165,892,970,3136,1524,4432,1188,3064,4291,2569,3551,
               4856,2441,3939,77,1417,262,198,2288,721,2220,3508,4229,3106,3822,2847,4341,431,2429,950,2859,3334,3084,
               3258,1617,4732,4267,106,2133,1226,1039,2522,3243,4416,2413,4069,513,1725,3837,1678,1822,3766,633,4224,
               2677,4707,1964,1339,4635,3917,301,2393,3269,4600,1817,3155,2642,3924,1993,310,2783,2235,1322,3919,4511,
               3810,4975,4564,2564,4908,4436,1588,2342,1753,1096,904,642,4739,4377,4192,4505,2887,2430,4358,593,4238,
               2563,260,2880,3843,4148,3989,4579,1966,4910,3520,3571,4829,3457,2416,2801,4966,1570,2653,3369,4519,1840,
               1247,2001,3249,2468,2408,3211,2668,4472,986,2082,2738,2632,3943,2424,101,3582,2043,1486,594,3828,946,
               1901,1258,1683,4010,2158,1455,506,1453,4042,1205,4276,2109,1175,4797,4029,1251,46,3858,3164,4677,2567,
               4155,4164,147,2968,2572,4045,4629,2073,1786,2084,4193,4053,1145,2241,4081,2190,312,3339,4067,3295,2410,
               4588,1192,3982,3365,4107,1369,4657,3218,2014,1557,4023,152,3000,4352,1611,3609,1381,3121,1639,2090,407,
               2053,55,3603,1677,987,1067,1375,2559,1304,4447,1335,4408,2674,1665,255,1279,2919,2147,2941,4205,1801,
               4110,915,4827,2521,4174,3033,335,551,3333,4935,4288,2114,1716,1942,3618,2029,718,3767,4668,140,1880,1097,
               1029,3130,2935,2918,4808,583,1841,91,930,907,3679,3167,1628,102,1866,4653,1884,4518,1416,1155,2326,2144,
               2233,2704,208,1193,3756,4631,724,1007,19,3198,2516,1316,54,3261,1859,2295,1088,4227,4330,4485,2313,1537,
               4020,3185,4135,750,3656,1126,2130,2170,4415,3744,2037,2979,2662,1350,2482,1568,3991,3671,1345,3420,4265,
               3926,1558,2737,1146,3117,2262,193,3820,38,640,4123,2698,3370,1076,2844,1401,4658,339,1528,2363,4061,2207,
               293,2366,1676,2765,1113,763,2238,4874,978,3597,4041,3423,4091,2168,1661,3795,4779,3709,811,3182,2582,
               4725,479,1227,3242,974,3101,1295,726,3588,1421,870,3406,3874,3914,3978,3112,4823,4230,1008,4688,1228,
               4050,2456,434,4055,2267,4906,3878,2320,2136,2113,2861,2914,2279,4780,2316,3641,2529,3615,561,4360,205,
               1682,4513,1516,4128,3476,4754,4374,2395,2656,3193,658,2638,1619,2291,3660,89,3181,644,760,1050,2225,823,
               4616,1234,4321,3192,1893,4752,754,1520,4407,707,143,785,2050,2680,4728,1864,2156,2862,3741,1121,171,412,
               3703,416,3415,3775,1294,4176,4723,3248,364,3401,2398,1439,740,73,4435,4491,1265,2208,2360,1988,4138,4543,
               2426,4946,1492,2202,1216,4477,4329,3545,2099,1565,2978,1998,3159,3972,2356,3049,631,2191,3013,4994,4713,
               1080,1405,632,1456,2035,3753,1851,4587,3003,3627,1802,2229,3670,178,498,2388,484,4261,4620,4459,2246,
               3613,3935,4401,877,1166,1689,1340,4250,273,3387,4810,759,2218,1597,1356,2821,3721,4539,1249,3094,1598,
               992,2093,3345,2515,3178,4068,1274,1843,4273,1466,3544,319,184,4167,4705,4125,4044,4234,3948,2028,839,
               3913,4591,2792,539,1154,3690,801,2245,3927,1135,3605,3470,210,4405,2512,65,863,374,4116,4934,1132,3907,
               3069,2557,1026,2328,341,4375,875,1506,3168,372,2624,2990,2259,2568,3523,4573,889,4433,4662,4221,4324,
               1325,2402,1579,96,4014,3250,3692,2637,4105,3737,4809,129,2561,2119,1635,1985,4920,4037,4470,3555,1398,
               4678,370,1788,2504,2773,2116,4898,4252,1820,994,881,737,231,377,1413,2910,2007,432,4264,3879,4361,1361,
               3042,4245,3483,1378,1591,1143,2643,172,3727,3863,1876,2654,3296,814,2256,2266,948,16,2026,2020,4355,367,
               1406,3176,2932,3367,3436,4750,4841,4025,3527,4693,4614,1848,4604,4983,4032,4380,1935,239,1624,3032,4476,
               542,2851,2725,2454,1459,1668,3867,1903,552,362,549,3018,4949,2086,2988,81,4026,2836,1730,4852,2925,355,
               3724,3607,3347,2936,2083,2449,2845,2690,4056,554,3004,693,1181,2834,3599,649,4521,1333,4285,420,2641,
               4998,4904,1229,1089,2780,2754,2746,4198,1534,2336,3946,3850,3492,2305,960,3313,1164,371,1306,2993,4145,
               3541,1575,3742,1484,3834,2548,4749,1858,258,3835,769,2061,4211,2018,3940,4550,3344,114,1816,1594,957,
               2973,2289,4309,3402,3818,679,2325,56,3300,1435,4480,435,3829,1057,3203,155,4216,2883,3409,1016,60,4019,
               2077,782,35,428,3494,4972,1650,3358,1060,358,1123,4956,4608,3708,3782,524,1200,2785,4422,265,3257,1585,
               1664,1968,3363,1518,4131,734,4218,2514,336,2596,4673,1156,887,3769,3020,494,580,1970,4210,4733,2479,1189,
               3277,2652,1133,212,4362,1064,3293,75,747,3796,1312,4646,1031,1004,247,1595,4734,3728,2159,1500,4049,3966,
               1919,186,1442,2015,3384,2056,1631,1307,1781,2422,1655,3153,1124,445,3716,4648,2503,4974,3047,1681,2800,
               3215,2581,243,3543,4479,2695,3549,1659,2333,1297,1674,2352,3995,611,1407,2474,3232,373,3062,3752,4574,
               4350,236,1344,2321,1218,1776,1498,4685,2097,510,1613,1623,2045,3163,3372,78,2317,4307,2874,3399,4762,911,
               3478,697,2386,363,1842,1153,1360,4626,3806,3142,4557,4950,2733,4991,4724,83,79,662,324,1440,4622,1130,
               2253,2376,808,1510,3105,2593,2576,3329,656,3070,858,1383,4528,4443,833,2927,1614,3797,4645,3356,213,2520,
               131,3087,4986,2524,3467,3800,448,3765,699,103,4606,523,3017,3851,1102,1821,1415,165,4640,3308,4253,3189,
               2555,3383,3962,2499,395,625,3997,947,761,3430,2739,1443,4039,645,1882,3802,4759,4147,2598,2541,4548,2822,
               4430,2286,4066,2827,4207,3268,183,3316,4057,1169,4426,4655,1711,4425,1178,2129,2532,397,1291,3661,3971,
               3458,1630,3096,4087,1289,3747,4279,901,3748,571,4997,4451,3366,2309,4327,4782,4767,1874,2742,3271,4921,
               2167,3437,4424,3525,4867,3115,3952,1937,1321,1125,2511,1527,2458,2495,3877,4747,1408,3272,3137,4789,2264,
               3749,2589,1502,2538,1320,1233,2236,1629,2789,123,2634,1796,3102,2710,765,1409,4534,2428,1745,3336,4746,
               4721,1873,2614,2138,798,4947,626,4553,1616,1782,2206,1890,790,7,2414,4853,2276,2960,4509,417,742,2127,
               3593,1909,824,4567,116,1481,1904,3216,4878,4504,3799,1024,592,4456,4627,2022,3980,1107,3825,2508,1797,
               1508,127,4793,2437,403,150,4909,2444,2442,62,2544,4191,3109,3557,4439,1576,3632,3590,3285,2153,4332,3676,
               222,4854,120,2380,3134,4289,2283,181,968,627,3330,4905,3454,1365,4006,614,3078,266,3841,3973,1932,2199,
               3190,1807,225,4201,4957,2068,2098,1186,2963,4038,2489,2186,4611,2870,2349,3730,284,3306,920,4517,2392,
               2354,4977,922,4786,1402,4858,4263,2469,2273,4108,1546,617,1092,4186,612,885,3444,777,1638,3252,4714,4880,
               28,3280,3604,3393,74,4078,3505,3143,2609,2227,2002,1584,1517,4212,3148,1072,2691,1799,2820,4242,2633,
               3958,3404,4764,3246,2684,115,4384,4862,4578,3702,3596,337,2687,4142,1973,753,1938,673,2712,3511,1201,
               4260,424,4421,3715,664,350,447,709,1924,1221,3922,2766,387,3547,1337,3205,1608,830,4726,3418,1734,4985,
               4943,1006,400,1253,21,1305,237,3107,3665,671,845,4565,1353,4835,3451,869,3686,604,3868,902,2904,1747,
               3310,2051,4047,3928,2757,4978,4417,696,1179,4832,2173,1090,848,1769,3808,3745,605,174,2269,2463,100,1243,
               741,334,2908,2881,3292,3463,2987,4423,4043,4812,1468,1838,3123,3674,460,1562,1975,3290,4515,738,1438,497,
               2750,3945,375,1209,404,3778,2467,93,3513,3204,2221,1849,827,4524,3187,4378,343,2683,2786,2547,3146,691,0,
               1582,1707,2021,3221,3591,2003,1131,4795,3717,2304,4736,2375,3395,2261,1731,1159,4342,2120,4757,2685,1671,
               4801,3273,1971,4461,2296,736,4122,4099,837,292,698,3969,4720,4937,1963,2274,3441,1868,2387,2671,3516,
               2502,3988,3999,63,1384,4333,650,1967,112,3024,804,577,126,3331,3793,1400,851,4760,3057,2665,4103,2995,
               2679,3893,2977,1053,4503,4969,2405,3860,153,2391,2353,2660,4729,280,4826,3821,3128,4496,1479,3811,504,
               903,2040,1672,3751,550,1862,1805,4907,945,3642,4259,1690,2107,4914,3125,3398,449,2903,941,1020,1532,1847,
               4011,3898,437,3445,739,2867,3291,586,2989,4444,1028,3279,3208,4917,573,4040,647,595,3396,1308,4566,4877,
               281,732,2763,2999,1908,3085,1111,4680,2195,160,2798,3165,952,4232,1331,1219,606,164,1040,3968,209,3612,
               329,2546,3929,4297,1429,433,860,4370,967,3729,3486,3734,3623,4971,3413,3695,2813,4652,908,2562,4605,4317,
               146,3361,1777,3186,508,4419,1561,4272,2852,168,3419,1392,2257,2237,136,894,197,1483,1792,4915,2850,3116,
               2389,1447,199,4525,3611,3466,1823,1795,730,3886,4393,2121,3465,807,2595,2997,1198,4403,821,2711,1241,
               4051,4887,623,4075,1162,2603,2775,3781,4387,1740,2282,3026,4163,24,216,2860,1910,4089,3421,4262,207,3510,
               2322,591,1720,1863,525,2335,469,2811,928,1896,2721,1469,2934,1913,956,1488,25,1134,2197,3224,1552,1645,
               2937,3533,2646,2187,4845,3854,2371,180,1434,3912,4703,3091,2344,3298,94,770,637,2536,2492,1208,275,1152,
               307,2059,4182,2610,4305,2787,1987,931,4226,3455,4651,2418,2431,201,2965,2023,4464,973,1634,2435,816,1954,
               3762,256,3678,1472,1605,560,4469,2957,2767,3299,1684,1167,3918,1204,4968,1071,4870,4420,2251,4180,2681,
               2956,1589,2659,2446,4187,430,3191,3942,2105,4009,4071,4353,3114,1641,4484,4369,756,3560,3622,1956,4958,
               2176,802,1207,2930,3531,2723,151,2292,714,544,4735,2745,3337,423,3219,659,797,755,2008,99,2496,1871,2046,
               4851,2163,2893,537,1701,2897,2222,2135,2599,705,3468,4373,1033,4859,4931,3264,1653,2607,2732]]


if __name__ == "__main__":
    dataset = SyntheticWL("test_swl", "exp")
    print(f"{dataset}\n  {dataset.data}")

    dataset = SyntheticWL("test_swl", "cexp")
    print(f"{dataset}\n  {dataset.data}")

    dataset = SyntheticWL("test_swl", "sr25")
    print(f"{dataset}\n  {dataset.data}")

    shutil.rmtree("test_swl")
