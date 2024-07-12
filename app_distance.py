from descriptor import glcm, bitdesc
from distances import manhattan, euclidean, chebyshev, canberra

path = 'image/XYZ/test.png'
patha = 'image/XYZ/testa.png'
pathb = 'image/XYZ/testb.png'


def main():
    
    feat_path = glcm(path)
    feat_patha = glcm(patha)
    feat_pathb = glcm(path)
    print(f'''\n\n        
                         Manhattan: {manhattan(feat_path, feat_patha)} | {manhattan(feat_path, feat_pathb)}\n
                         Euclidean: {euclidean(feat_path, feat_patha)} | {euclidean(feat_path, feat_pathb)}\n
                         Chebyshev: {chebyshev(feat_path, feat_patha)} | {chebyshev(feat_path, feat_pathb)}\n
                         Canberra: {canberra(feat_path, feat_patha)} | {canberra(feat_path, feat_pathb)}\n
                            ''')

if __name__ == '__main__':
    main()