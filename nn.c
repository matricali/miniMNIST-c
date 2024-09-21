#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.001f
#define EPOCHS 20
#define BATCH_SIZE 64
#define IMAGE_SIZE 28
#define TRAIN_SPLIT 0.8
#define NUM_THREADS 8

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"

typedef struct {
    float *weights, *biases;
    int input_size, output_size;
} Layer;

typedef struct {
    Layer hidden, output;
} Network;

typedef struct {
    unsigned char *images, *labels;
    int nImages;
} InputData;

typedef struct {
    Network *net;
    InputData *data;
    int start_idx, end_idx;
    float learning_rate;
    float *total_loss;
    int *correct_preds;
    pthread_mutex_t *loss_mutex;
    pthread_mutex_t *preds_mutex;
} ThreadData;

void softmax(float *input, int size) {
    float max = input[0], sum = 0;
    for (int i = 1; i < size; i++)
        if (input[i] > max) max = input[i];
    for (int i = 0; i < size; i++) {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }
    for (int i = 0; i < size; i++)
        input[i] /= sum;
}

void init_layer(Layer *layer, int in_size, int out_size) {
    int n = in_size * out_size;
    float scale = sqrtf(2.0f / in_size);

    layer->input_size = in_size;
    layer->output_size = out_size;
    layer->weights = malloc(n * sizeof(float));
    layer->biases = calloc(out_size, sizeof(float));

    for (int i = 0; i < n; i++)
        layer->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
}

void forward(Layer *layer, float *input, float *output) {
    for (int i = 0; i < layer->output_size; i++) {
        output[i] = layer->biases[i];
        for (int j = 0; j < layer->input_size; j++)
            output[i] += input[j] * layer->weights[j * layer->output_size + i];
    }
}

void backward(Layer *layer, float *input, float *output_grad, float *input_grad, float lr) {
    for (int i = 0; i < layer->output_size; i++) {
        for (int j = 0; j < layer->input_size; j++) {
            int idx = j * layer->output_size + i;
            float grad = output_grad[i] * input[j];
            layer->weights[idx] -= lr * grad;
            if (input_grad)
                input_grad[j] += output_grad[i] * layer->weights[idx];
        }
        layer->biases[i] -= lr * output_grad[i];
    }
}

void train(Network *net, float *input, int label, float lr, float *loss, int *correct) {
    float hidden_output[HIDDEN_SIZE], final_output[OUTPUT_SIZE];
    float output_grad[OUTPUT_SIZE] = {0}, hidden_grad[HIDDEN_SIZE] = {0};

    forward(&net->hidden, input, hidden_output);
    for (int i = 0; i < HIDDEN_SIZE; i++)
        hidden_output[i] = hidden_output[i] > 0 ? hidden_output[i] : 0;  // ReLU

    forward(&net->output, hidden_output, final_output);
    softmax(final_output, OUTPUT_SIZE);

    // Calculation of loss and accuracy
    *loss += -logf(final_output[label]);
    int predicted_label = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (final_output[i] > final_output[predicted_label])
            predicted_label = i;
    }
    if (predicted_label == label) {
        (*correct)++;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++)
        output_grad[i] = final_output[i] - (i == label);

    backward(&net->output, hidden_output, output_grad, hidden_grad, lr);

    for (int i = 0; i < HIDDEN_SIZE; i++)
        hidden_grad[i] *= hidden_output[i] > 0 ? 1 : 0;  // ReLU derivative

    backward(&net->hidden, input, hidden_grad, NULL, lr);
}

int predict(Network *net, float *input) {
    float hidden_output[HIDDEN_SIZE], final_output[OUTPUT_SIZE];

    forward(&net->hidden, input, hidden_output);
    for (int i = 0; i < HIDDEN_SIZE; i++)
        hidden_output[i] = hidden_output[i] > 0 ? hidden_output[i] : 0;  // ReLU

    forward(&net->output, hidden_output, final_output);
    softmax(final_output, OUTPUT_SIZE);

    int max_index = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++)
        if (final_output[i] > final_output[max_index])
            max_index = i;

    return max_index;
}

void *train_batch(void *arg) {
    ThreadData *td = (ThreadData *)arg;
    float img[INPUT_SIZE];
    float batch_loss = 0;
    int batch_correct = 0;

    for (int i = td->start_idx; i < td->end_idx; i++) {
        for (int k = 0; k < INPUT_SIZE; k++)
            img[k] = td->data->images[i * INPUT_SIZE + k] / 255.0f;
        train(td->net, img, td->data->labels[i], td->learning_rate, &batch_loss, &batch_correct);
    }

    pthread_mutex_lock(td->loss_mutex);
    *(td->total_loss) += batch_loss;
    pthread_mutex_unlock(td->loss_mutex);

    pthread_mutex_lock(td->preds_mutex);
    *(td->correct_preds) += batch_correct;
    pthread_mutex_unlock(td->preds_mutex);

    return NULL;
}

void read_mnist_images(const char *filename, unsigned char **images, int *nImages) {
    FILE *file = fopen(filename, "rb");
    if (!file) exit(1);

    int temp, rows, cols;
    fread(&temp, sizeof(int), 1, file);
    fread(nImages, sizeof(int), 1, file);
    *nImages = __builtin_bswap32(*nImages);

    fread(&rows, sizeof(int), 1, file);
    fread(&cols, sizeof(int), 1, file);

    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    *images = malloc((*nImages) * IMAGE_SIZE * IMAGE_SIZE);
    fread(*images, sizeof(unsigned char), (*nImages) * IMAGE_SIZE * IMAGE_SIZE, file);
    fclose(file);
}

void read_mnist_labels(const char *filename, unsigned char **labels, int *nLabels) {
    FILE *file = fopen(filename, "rb");
    if (!file) exit(1);

    int temp;
    fread(&temp, sizeof(int), 1, file);
    fread(nLabels, sizeof(int), 1, file);
    *nLabels = __builtin_bswap32(*nLabels);

    *labels = malloc(*nLabels);
    fread(*labels, sizeof(unsigned char), *nLabels, file);
    fclose(file);
}

void shuffle_data(unsigned char *images, unsigned char *labels, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        for (int k = 0; k < INPUT_SIZE; k++) {
            unsigned char temp = images[i * INPUT_SIZE + k];
            images[i * INPUT_SIZE + k] = images[j * INPUT_SIZE + k];
            images[j * INPUT_SIZE + k] = temp;
        }
        unsigned char temp = labels[i];
        labels[i] = labels[j];
        labels[j] = temp;
    }
}

int main() {
    Network net;
    InputData data = {0};
    float learning_rate = LEARNING_RATE;

    srand(time(NULL));

    init_layer(&net.hidden, INPUT_SIZE, HIDDEN_SIZE);
    init_layer(&net.output, HIDDEN_SIZE, OUTPUT_SIZE);

    read_mnist_images(TRAIN_IMG_PATH, &data.images, &data.nImages);
    read_mnist_labels(TRAIN_LBL_PATH, &data.labels, &data.nImages);

    shuffle_data(data.images, data.labels, data.nImages);

    int train_size = (int)(data.nImages * TRAIN_SPLIT);
    int test_size = data.nImages - train_size;

    pthread_mutex_t loss_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t preds_mutex = PTHREAD_MUTEX_INITIALIZER;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0;
        int correct_preds = 0;
        pthread_t threads[NUM_THREADS];
        ThreadData thread_data[NUM_THREADS];

        int batch_size = train_size / NUM_THREADS;

        for (int t = 0; t < NUM_THREADS; t++) {
            thread_data[t].net = &net;
            thread_data[t].data = &data;
            thread_data[t].start_idx = t * batch_size;
            thread_data[t].end_idx = (t + 1) * batch_size;
            thread_data[t].learning_rate = learning_rate;
            thread_data[t].total_loss = &total_loss;
            thread_data[t].correct_preds = &correct_preds;
            thread_data[t].loss_mutex = &loss_mutex;
            thread_data[t].preds_mutex = &preds_mutex;

            pthread_create(&threads[t], NULL, train_batch, &thread_data[t]);
        }

        for (int t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }

        float accuracy = (float)correct_preds / train_size;
        float average_loss = total_loss / train_size;
        printf("Epoch %d/%d, Accuracy: %.2f%%, Loss: %.4f\n", epoch + 1, EPOCHS, accuracy * 100, average_loss);
    }

    free(net.hidden.weights);
    free(net.hidden.biases);
    free(net.output.weights);
    free(net.output.biases);
    free(data.images);
    free(data.labels);

    return 0;
}
