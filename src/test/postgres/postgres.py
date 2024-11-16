import psycopg2

dsn = "dbname=postgres host=localhost user=postgres password=y6KDgfg9"

with psycopg2.connect(dsn) as conn:
    with conn.cursor() as cur:
        sql = "select version()"
        cur.execute(sql)
        print(cur.fetchone()[0])
        print("========== ========== ==========")
        sql = "SELECT * FROM items;"
        cur.execute(sql)
        ret = cur.fetchall()
        for i in range(len(ret)):
            print(ret[i])
        print("========== ========== ==========")
        vec = "[3.1,-1.5,2.8]"
        sql = f"SELECT *, embedding <-> '{vec}' AS l2_distance FROM items ORDER BY l2_distance;"
        cur.execute(sql)
        ret = cur.fetchall()
        for i in range(len(ret)):
            print(ret[i])
        print("========== ========== ==========")
        sql = f"SELECT *, (embedding <#> '{vec}') * -1 AS inner_product FROM items ORDER BY inner_product;"
        cur.execute(sql)
        ret = cur.fetchall()
        for i in range(len(ret)):
            print(ret[i])
        print("========== ========== ==========")
        sql = f"SELECT *, 1 - (embedding <=> '{vec}') AS cosine_similarity FROM items ORDER BY cosine_similarity;"
        cur.execute(sql)
        ret = cur.fetchall()
        for i in range(len(ret)):
            print(ret[i])
        print("========== ========== ==========")
