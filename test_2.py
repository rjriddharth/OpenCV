import cv2
import argparse
import numpy as np

def help():
    print("\n"
          "This program demonstrates a method for shape comparison based on Shape Context\n"
          "You should run the program providing a number between 1 and 20 for selecting an image in the folder ../data/shape_sample.\n"
          "Call\n"
          "./shape_example [number between 1 and 20, 1 default]\n\n")

def simple_contour(current_query, n=300):
    contours_query, _ = cv2.findContours(current_query, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_query = [point for border in contours_query for point in border]
    
    # In case actual number of points is less than n
    dummy = 0
    for add in range(len(contours_query), n):
        contours_query.append(contours_query[dummy])  # adding dummy values
    
    # Uniformly sampling
    np.random.shuffle(contours_query)
    return contours_query[:n]

def main():
    path = "data/shape_sample/"
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=int, default=1, help='Index of the input image (between 1 and 20)')
    args = parser.parse_args()

    if args.input < 1 or args.input > 20:
        help()
        return

    mysc = cv2.createShapeContextDistanceExtractor()
    sz2Sh = (300, 300)
    query_name = f"{path}{args.input}.png"
    query = cv2.imread(query_name, cv2.IMREAD_GRAYSCALE)
    query_to_show = cv2.resize(query, sz2Sh, interpolation=cv2.INTER_LINEAR_EXACT)
    cv2.imshow("QUERY", query_to_show)
    cv2.moveWindow("TEST", 0, 0)
    cont_query = simple_contour(query)

    best_match = 0
    best_dis = float('inf')
    for ii in range(1, 21):
        if ii == args.input:
            continue
        cv2.waitKey(30)
        iiname = f"{path}{ii}.png"
        print(f"name: {iiname}")
        ii_im = cv2.imread(iiname, 0)
        ii_to_show = cv2.resize(ii_im, sz2Sh, interpolation=cv2.INTER_LINEAR_EXACT)
        cv2.imshow("TEST", ii_to_show)
        cv2.moveWindow("TEST", sz2Sh[0] + 50, 0)
        cont_ii = simple_contour(ii_im)
        dis = mysc.computeDistance(cont_query, cont_ii)
        if dis < best_dis:
            best_match = ii
            best_dis = dis
        print(f"distance between {query_name} and {iiname} is: {dis}")

    cv2.destroyWindow("TEST")
    best_name = f"{path}{best_match}.png"
    ii_im = cv2.imread(best_name, 0)
    best_to_show = cv2.resize(ii_im, sz2Sh, interpolation=cv2.INTER_LINEAR_EXACT)
    cv2.imshow("BEST MATCH", best_to_show)
    cv2.moveWindow("BEST MATCH", sz2Sh[0] + 50, 0)
    cv2.waitKey()

if __name__ == "__main__":
    main()
