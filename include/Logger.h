//
// Created by termed on 30.05.2026.
//

#ifndef NEURALNETWORKS_LOGGER_H
#define NEURALNETWORKS_LOGGER_H
#include <iosfwd>
#include <iostream>


/**
 * @class Logger
 * @brief A lightweight, conditional logging utility for streaming runtime info.
 * * This class provides formatted console logging based on an enum prefix. It also
 * implements a "blackhole" stream via NullBuffer, silencing all logging outputs
 * completely when verbose mode is disabled without needing cluttered if-statements
 * at the call site.
 */
class Logger {
public:

    bool verbose_ = false; ///< Flag determining whether logs are printed to standard output or suppressed.

    /**
     * @enum Prefix
     * @brief System sub-categories used to tag and prefix individual log messages.
     */
    enum Prefix {
        TRAINING, ///< Prefix string for training milestones and cost updates: `[TRAINING] `
        DATASET,  ///< Prefix string for data loading or transformations: `[DATASET] `
        TESTING,  ///< Prefix string for accuracy scores and evaluations: `[TESTING] `
        OUTPUT    ///< Prefix string for raw model outputs or predictions: `[OUTPUT] `
    };

    /**
     * @class NullBuffer
     * @brief A custom stream buffer that discards all incoming data characters.
     * * Overrides standard stream behavior to create an efficient, non-allocating
     * sink stream for logging suppression.
     */
    class NullBuffer : public std::streambuf {
    public:
        /**
         * @brief Overflows the buffer by simply swallowing characters.
         * @param c The character pushed into the stream.
         * @return The passed character value to indicate success.
         */
        int overflow(int c) override {
            return c;
        }
    };

    /**
     * @brief Constructs a Logger instance.
     * @param verbose Toggles whether to display console output (`true`) or swallow it (`false`).
     */
    Logger(bool verbose) : verbose_(verbose), null_stream_(&null_buffer_) {}

    /**
     * @brief Converts a given Prefix enum value to its string literal equivalent.
     * @param p The prefix enum variant.
     * @return A compile-time std::string_view containing the bracketed tag.
     */
    static constexpr std::string_view prefix(Prefix p) {
        switch (p){
            case TRAINING:  return "[ TRAINING ] ";
            case TESTING:   return "[ TESTING ] ";
            case DATASET:   return "[ DATASET ] ";
            case OUTPUT:    return "[ OUTPUT ] ";
            default:        return "[ GENERAL ] ";
        }
    }

    /**
     * @brief Initiates a log entry line.
     * * Conditionally redirects outputs. If #verbose_ is true, it prints out the
     * associated tag prefix and redirects data to `std::cout`. If false, data is
     * redirected invisibly to the NullBuffer.
     * * @param p The Prefix category marking this log.
     * @return Reference to a standard output stream object (`std::ostream&`) ready for `<<` chaining.
     */
    std::ostream& log(Prefix p) {
        if (verbose_) {
            return std::cout << prefix(p);
        }
        return null_stream_;
    }

private:
    NullBuffer null_buffer_;   ///< Internal custom stream buffer that consumes characters without saving them.
    std::ostream null_stream_; ///< Deployed fallback stream used when verbose tracking is turned off.
};

#endif //NEURALNETWORKS_LOGGER_H