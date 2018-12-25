import itertools as it
import pandas as pd
import http.client
import json


def set_connection(clientid, clientsecret):
    conn = http.client.HTTPSConnection("api.citivelocity.com")
    payload = "grant_type=client_credentials&client_id=" + clientid + "&client_secret=" + clientsecret + "&scope=/api"
    headers = {
        'content-type': "application/x-www-form-urlencoded",
        'accept': "application/json"
    }
    conn.request("POST", "/markets/cv/api/oauth2/token", payload, headers)
    res = conn.getresponse()
    data = res.read()
    my_json = data.decode('utf8').replace("'", '"')
    jsondata = json.loads(my_json)
    return (jsondata['access_token'])


def get_data(clientId, access_token, startDate, endDate, ticker):
    # access_token = set_connection(clientId, clientSecret)
    conn = http.client.HTTPSConnection("api.citivelocity.com")
    payload = "{\"startDate\": " + startDate + ", \"endDate\": " + endDate + ", \"tags\":[\"" + ticker + "\"]}"
    headers = {
        'content-type': "application/json",
        'accept': "application/json",
        'authorization': "Bearer " + access_token
    }
    conn.request("POST",
                 "/markets/analytics/chartingbe/rest/external/authed/data?client_id=" + clientId,
                 payload, headers)
    res = conn.getresponse()
    data = res.read()
    print(data)
    my_json = data.decode('utf8').replace("'", '"')
    jsondata = json.loads(my_json)
    df = pd.DataFrame(dict(date=jsondata['body'][ticker]['x'], value=jsondata['body'][ticker]['c']))

    return df


def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def save_data(df, file_name):
    df.to_csv(file_name, index=False, sep=",")
    return

    '''
def main():
    set_connection("2596265e-2930-46c8-95ac-f4503c7fb556","rQ5bO6dJ6xT3dN5dL1oD4dN3cG0lL2hU3aG1jG1wN8fN2bW7qI")
    item_template = 'cred_ix_{}_{}_{}'
    attr = 'spd_vol'
    cut = 'nyclose'
    source = 'citi'
    tag_template = 'citi_template:{}_{}_{}'
    expiries = ['1m', '2m', '3m', '6m', '1y', '2y']
    indices = ['hy', 'ig']
    moneyness = ['atm']

    d = {tag_template.format(i, e, m): (item_template.format(i, e, m), attr, cut, source) for i, e, m in
         it.product(expiries, indices, moneyness)}
    d.keys()



    set_connection("2596265e-2930-46c8-95ac-f4503c7fb556", "rQ5bO6dJ6xT3dN5dL1oD4dN3cG0lL2hU3aG1jG1wN8fN2bW7qI")
    df = get_data("2596265e-2930-46c8-95ac-f4503c7fb556", "rQ5bO6dJ6xT3dN5dL1oD4dN3cG0lL2hU3aG1jG1wN8fN2bW7qI",
                  "20170101", "20180101", "EQUITY.EQIVOL.EQUITY_INDEX.92141.EQIVOL_FWD.STRIKE_ATM.2M.CITI")

    print(df)


if __name__ == '__main__':
    main()

    '''