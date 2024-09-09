#include <iostream>
#include <string>
using namespace std;

int main(int argc, char* argv[]) {

    // Read command line arguments
    if (argc < 2) {
        cerr << "Error: missing file argument" << endl;
        cerr << "Usage " << argv[0] << " <input_file>" << endl;
        return 1;
    }
    string inputName(argv[1]);
    cout << "Adding Lund Jet Plane info to " << inputName << endl;

}