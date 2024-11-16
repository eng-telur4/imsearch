def vec2str(vecs) -> list:
    size = int(vecs[0].shape[0])
    vecs_s = []
    for vec in vecs:
        s = "["
        for idx, scala in enumerate(vec):
            s += str(scala)
            if idx != size-1:
                s += ","
        s += "]"
        vecs_s.append(s)
    return vecs_s